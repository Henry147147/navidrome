"""
Hosts the server that will create the embedding from the music file and push it to milvus client
"""

import json
import logging
import os
import shlex
import socket
from hashlib import sha224
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from threading import Thread
from time import sleep
from typing import Iterable, List, Optional, Tuple

import librosa

import numpy as np
import torch
import torchaudio
from pymilvus import MilvusClient
from muq import MuQMuLan
from cueparser import CueSheet
from models import TrackSegment, ChunkedEmbedding, SongEmbedding
from upload_features import UploadFeaturePipeline, UploadSettings
from database_query import MilvusSimilaritySearcher


SAMPLE_RATE = 24_000
WINDOW_SECONDS = 2 * 60
HOP_SECONDS = 15
STORAGE_DTYPE = torch.float32
MODEL_ID = "OpenMuQ/MuQ-MuLan-large"
DEVICE = "cuda"
SOCKET_PATH = "/tmp/navidrome_embed.sock"


logger = logging.getLogger("navidrome.embed_server")


def cue_time_to_seconds(time_str: str) -> float:
    minute, second, frame = time_str.split(":")
    return int(minute) * 60 + int(second) + int(frame) / 75.0


def parse_cuesheet_tracks(
    cue_path: Path,
    music_file: str,
    *,
    candidate_names: Optional[Iterable[str]] = None,
) -> List[TrackSegment]:
    target_path = Path(music_file)
    target_name = target_path.name.lower()
    target_stem = target_path.stem.lower()
    candidate_name_set = {target_name}
    candidate_stem_set = {target_stem}
    if candidate_names:
        for name in candidate_names:
            if not name:
                continue
            candidate_name_set.add(Path(name).name.lower())
            candidate_stem_set.add(Path(name).stem.lower())
    logger.debug(
        "Parsing cuesheet %s for target file %s (candidates=%s)",
        cue_path,
        target_name,
        sorted(candidate_name_set),
    )
    try:
        raw_text = cue_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.debug("Failed to decode %s as UTF-8, retrying with cp1252", cue_path)
        raw_text = cue_path.read_text(encoding="cp1252", errors="ignore")
    normalized = raw_text.lstrip("\ufeff")
    if normalized != raw_text:
        logger.debug("Removed UTF-8 BOM from cuesheet %s", cue_path)
    lines = normalized.splitlines()
    removed_leading = 0
    while lines and not lines[0].strip():
        lines.pop(0)
        removed_leading += 1
    if removed_leading:
        logger.debug(
            "Stripped %d leading blank lines from cuesheet %s",
            removed_leading,
            cue_path,
        )
    if not lines:
        logger.debug("Cuesheet %s contains no data after normalization", cue_path)
        return []
    cuesheet = "\n".join(lines)
    sheet = CueSheet()
    sheet.setOutputFormat("", "%title%")
    sheet.setData(cuesheet)
    try:
        sheet.parse()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to parse cuesheet %s: %s", cue_path, exc)
        return []
    logger.debug(
        "cueparser extracted %d tracks for %s (sheet file=%s)",
        len(sheet.tracks),
        target_name,
        (sheet.file or "").lower(),
    )

    track_contexts = []
    current_file: Optional[str] = None
    current_track: Optional[dict] = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        try:
            parts = shlex.split(line, comments=False, posix=True)
        except ValueError:
            continue
        if not parts:
            continue
        keyword = parts[0].upper()
        if keyword == "FILE" and len(parts) >= 2:
            current_file = Path(parts[1]).name.lower()
        elif keyword == "TRACK" and len(parts) >= 2:
            try:
                track_number = int(parts[1])
            except ValueError:
                track_number = None
            current_track = {"number": track_number, "file": current_file}
            track_contexts.append(current_track)
        elif (
            keyword == "TITLE"
            and len(parts) >= 2
            and current_track is not None
            and current_track.get("title") is None
        ):
            current_track["title"] = parts[1]

    if len(track_contexts) != len(sheet.tracks):
        logger.debug(
            "Cuesheet %s track count mismatch: parsed %d tracks, context %d",
            cue_path,
            len(sheet.tracks),
            len(track_contexts),
        )

    collected: List[dict] = []
    for track, context in zip(sheet.tracks, track_contexts):
        context_file = context.get("file")
        if context_file:
            context_name = Path(context_file).name.lower()
            context_stem = Path(context_file).stem.lower()
            if (
                context_name not in candidate_name_set
                and context_stem not in candidate_stem_set
            ):
                logger.debug(
                    "Skipping track %s due to file mismatch (context=%s, targets=%s)",
                    context.get("number") or track.number,
                    context_file,
                    sorted(candidate_name_set),
                )
                continue
        if not track.offset:
            logger.debug(
                "Skipping track %s because no offset present in cuesheet",
                context.get("number") or track.number,
            )
            continue
        try:
            start = cue_time_to_seconds(track.offset)
        except ValueError:
            logger.debug(
                "Skipping track %s due to invalid offset format: %s",
                context.get("number") or track.number,
                track.offset,
            )
            continue
        title = track.title or context.get("title")
        context_number = context.get("number")
        number = context_number if context_number is not None else track.number
        collected.append(
            {
                "number": int(number) if number is not None else track.number,
                "title": title,
                "start": start,
            }
        )

    collected.sort(key=lambda entry: entry["start"])
    result: List[TrackSegment] = []
    for idx, entry in enumerate(collected):
        end = collected[idx + 1]["start"] if idx + 1 < len(collected) else None
        title = entry["title"] or f"Track {entry['number']:02d}"
        result.append(
            TrackSegment(
                index=entry["number"],
                title=title,
                start=entry["start"],
                end=end,
            )
        )
    logger.debug("Parsed %d cue tracks for %s", len(result), target_name)
    return result


def load_audio_segment(
    music_file: str,
    *,
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> np.ndarray:
    logger.debug(
        "Loading audio segment from %s (offset=%s, duration=%s)",
        music_file,
        offset,
        duration,
    )
    info_fn = getattr(torchaudio, "info", None)
    if info_fn is None:
        logger.debug("torchaudio.info unavailable, falling back to librosa")
        return _load_audio_with_librosa(music_file, offset=offset, duration=duration)

    try:
        info = info_fn(str(music_file))
        source_sr = info.sample_rate
    except Exception:
        logger.exception(
            "Unable to inspect %s with torchaudio, using librosa", music_file
        )
        return _load_audio_with_librosa(music_file, offset=offset, duration=duration)

    frame_offset = max(int(offset * source_sr), 0)
    if duration is not None:
        num_frames = max(int(duration * source_sr), 0)
    else:
        num_frames = -1

    try:
        waveform, sr = torchaudio.load(
            str(music_file),
            frame_offset=frame_offset,
            num_frames=num_frames,
        )
    except Exception:
        logger.exception("torchaudio.load failed for %s, using librosa", music_file)
        return _load_audio_with_librosa(music_file, offset=offset, duration=duration)

    if waveform.numel() == 0:
        return np.zeros(0, dtype=np.float32)

    waveform = waveform.to(STORAGE_DTYPE)
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)

    if sr != SAMPLE_RATE:
        logger.debug("Resampling audio from %s Hz to %s Hz", sr, SAMPLE_RATE)
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    logger.debug(
        "Loaded audio segment with %d samples (channels collapsed)", waveform.numel()
    )
    return waveform.contiguous().cpu().numpy()


def _load_audio_with_librosa(
    music_file: str, offset: float, duration: Optional[float]
) -> np.ndarray:
    safe_offset = max(float(offset), 0.0)
    safe_duration = None if duration is None else max(float(duration), 0.0)
    logger.debug(
        "Loading audio with librosa from %s (offset=%s, duration=%s)",
        music_file,
        safe_offset,
        safe_duration,
    )
    audio, _ = librosa.load(
        music_file,
        sr=SAMPLE_RATE,
        mono=True,
        offset=safe_offset,
        duration=safe_duration,
    )
    if audio.size == 0:
        logger.warning("Loaded empty audio buffer for %s", music_file)
        return np.zeros(0, dtype=np.float32)
    logger.debug("Loaded %d samples using librosa", audio.size)
    return audio.astype(np.float32, copy=False)


class ClassifierModel:
    def __init__(self, timeout=360):
        self._last_used = datetime.now(UTC)
        self._model = None
        self.timeout = timeout
        self.timeout_thread = self.model_unloader()
        self.timeout_thread.start()
        logger.debug(
            "ClassifierModel initialized with timeout %s seconds", self.timeout
        )

    def _model_used_now(self):
        self._last_used = datetime.now(UTC)
        logger.debug("Model usage timestamp updated to %s", self._last_used)

    def model_unloader(self):
        def _run():
            while True:
                if self._model is None:
                    sleep(self.timeout)
                else:
                    delta = datetime.now(UTC) - self._last_used
                    if delta.seconds > self.timeout:
                        logger.info(
                            "Model idle for %s seconds, releasing from memory", delta
                        )
                        del self._model
                        self._model = None
                        try:
                            torch.cuda.empty_cache()
                            logger.debug("torch.cuda cache cleared")
                        except:
                            pass
                sleep(5)

        return Thread(target=_run)

    def ensure_model_loaded(self):
        if self._model is None:
            logger.info("Loading MuQMuLan model %s onto %s", MODEL_ID, DEVICE)
            self._model = (
                MuQMuLan.from_pretrained(MODEL_ID).to(DEVICE).to(STORAGE_DTYPE).eval()
            )
        else:
            logger.debug("Model already loaded; skipping reload")
        self._model_used_now()

    def run_inference(
        self,
        music_file: str,
        music_name: str,
        cue_file: Optional[str],
    ):
        logger.info(
            "Running inference for %s (name=%s, cue=%s)",
            music_file,
            music_name,
            cue_file,
        )
        if self._model is None:
            raise RuntimeError("Model is not loaded. Call ensure_model_loaded() first.")

        self._model_used_now()

        music_path = Path(music_file)
        cue_path: Optional[Path] = Path(cue_file) if cue_file is not None else None

        segments: List[TrackSegment] = []
        if cue_path is not None:
            logger.debug("Attempting to load cuesheet %s", cue_path)
            segments = parse_cuesheet_tracks(
                cue_path,
                str(music_path),
                candidate_names=[music_name],
            )

        if not segments:
            logger.debug("No cue segments detected; processing entire track")
            segments = [
                TrackSegment(index=1, title=music_name, start=0.0, end=None),
            ]
        logger.debug("Prepared %d segments for inference", len(segments))

        segment_payloads = []
        param_iter = iter(self._model.parameters())
        first_param = next(param_iter, None)
        if first_param is not None:
            model_device = first_param.device
        else:
            model_device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        logger.debug("Using model device %s", model_device)
        empty_cache = (
            torch.cuda.empty_cache if model_device.type == "cuda" else (lambda: None)
        )

        chunk_size = int(WINDOW_SECONDS * SAMPLE_RATE)
        hop_size = int(HOP_SECONDS * SAMPLE_RATE)
        if chunk_size <= 0 or hop_size <= 0:
            raise ValueError("Invalid chunk or hop size configuration.")

        for segment in segments:
            logger.info(
                "Embedding segment %s (%s) starting at %ss",
                segment.index,
                segment.title,
                segment.start,
            )
            offset = float(segment.start)
            duration = segment.duration
            audio = load_audio_segment(
                str(music_path),
                offset=offset,
                duration=duration,
            )
            if audio.size == 0:
                logger.warning(
                    "Segment %s produced empty audio buffer; skipping", segment.index
                )
                continue

            total_samples = audio.shape[0]
            starts = list(range(0, max(total_samples - chunk_size, 0) + 1, hop_size))
            if not starts or (starts[-1] + chunk_size < total_samples):
                starts.append(max(total_samples - chunk_size, 0))

            chunk_arrays = []
            chunk_metadata = []
            for idx, start_sample in enumerate(starts):
                end_sample = min(start_sample + chunk_size, total_samples)
                chunk = audio[start_sample:end_sample]
                observed = int(chunk.shape[0])
                if observed == 0:
                    continue
                if observed < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - observed))
                chunk_arrays.append(chunk.astype("float32", copy=False))
                chunk_metadata.append(
                    {
                        "index": idx,
                        "start_seconds": offset + start_sample / SAMPLE_RATE,
                        "end_seconds": offset + end_sample / SAMPLE_RATE,
                    }
                )

            if not chunk_arrays:
                logger.warning("No audio chunks produced for segment %s", segment.index)
                continue

            chunk_matrix = np.stack(chunk_arrays, axis=0)
            chunk_tensor = (
                torch.from_numpy(chunk_matrix).to(model_device).to(STORAGE_DTYPE)
            )

            with torch.inference_mode():
                segment_embeddings = self._model(wavs=chunk_tensor)

            segment_embeddings = torch.nn.functional.normalize(
                segment_embeddings, dim=1
            )
            centroid = torch.nn.functional.normalize(
                segment_embeddings.mean(dim=0), dim=0
            )

            centroid_cpu = centroid.detach().to("cpu", dtype=STORAGE_DTYPE)
            chunk_cpu = segment_embeddings.detach().to("cpu", dtype=STORAGE_DTYPE)

            logger.debug(
                "Segment %s produced %d chunks", segment.index, chunk_cpu.shape[0]
            )

            computed_duration: Optional[float] = (
                float(duration)
                if duration is not None
                else (
                    chunk_metadata[-1]["end_seconds"]
                    - chunk_metadata[0]["start_seconds"]
                    if chunk_metadata
                    else None
                )
            )

            segment_payloads.append(
                {
                    "index": segment.index,
                    "title": segment.title,
                    "offset_seconds": offset,
                    "duration_seconds": computed_duration,
                    "num_chunks": int(chunk_cpu.shape[0]),
                    "centroid": centroid_cpu.tolist(),
                    "chunk_embeddings": chunk_cpu.tolist(),
                    "chunk_metadata": chunk_metadata,
                }
            )

            del chunk_tensor, segment_embeddings
            empty_cache()
            logger.debug("Cleared temporary tensors for segment %s", segment.index)

        if not segment_payloads:
            raise RuntimeError("Unable to generate embeddings for the provided audio.")

        logger.info(
            "Generated embeddings for %d segments of %s",
            len(segment_payloads),
            music_file,
        )
        return {
            "music_file": str(music_path),
            "cue_file": str(cue_path) if cue_path is not None else None,
            "model_id": MODEL_ID,
            "sample_rate": SAMPLE_RATE,
            "window_seconds": WINDOW_SECONDS,
            "hop_seconds": HOP_SECONDS,
            "generated_at": datetime.now(UTC).isoformat(),
            "segments": segment_payloads,
        }

    def embed_music(
        self, music_file: str, music_name: str, cue_file: Optional[str] = None
    ):
        self.ensure_model_loaded()
        return self.run_inference(music_file, music_name, cue_file)


class EmbedSocketServer:
    def __init__(
        self,
        socket_path: str = SOCKET_PATH,
        *,
        milvus_client: Optional[MilvusClient] = None,
    ):
        self.socket_path = socket_path
        self.logger = logger
        self.model = ClassifierModel()
        self.milvus_client = milvus_client or MilvusClient("http://localhost:19530")
        self.similarity_searcher = MilvusSimilaritySearcher(
            self.milvus_client,
            logger=self.logger,
        )
        self.feature_pipeline = UploadFeaturePipeline(
            similarity_searcher=self.similarity_searcher,
            logger=self.logger,
        )
        self.logger.debug(
            "EmbedSocketServer initialized for %s",
            self.socket_path,
        )

    def serve_forever(self) -> None:
        if os.path.exists(self.socket_path):
            self.logger.debug("Removing existing socket at %s", self.socket_path)
            os.unlink(self.socket_path)

        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server_sock.bind(self.socket_path)
            os.chmod(self.socket_path, 0o600)
            server_sock.listen(1)
            self.logger.info("Embedding server listening on %s", self.socket_path)

            while True:
                self.logger.debug("Waiting for incoming connection")
                conn, _ = server_sock.accept()
                self.logger.debug("Accepted new connection")
                try:
                    self.handle_connection(conn)
                except Exception:  # pragma: no cover - defensive logging
                    self.logger.exception("Unexpected error handling connection")
        finally:
            server_sock.close()
            self.logger.debug("Server socket closed")
            try:
                os.unlink(self.socket_path)
            except FileNotFoundError:
                pass

    @staticmethod
    def load_from_json(
        data: dict,
    ) -> Tuple[List[SongEmbedding], List[ChunkedEmbedding]]:
        window_seconds = data["window_seconds"]
        hop_seconds = data["hop_seconds"]
        sample_rate = data["sample_rate"]
        song_tracks = data["segments"]
        chunk_data = []
        song_data = []

        for song in song_tracks:
            name = song["title"]
            index = song["index"]
            start_seconds = song["offset_seconds"]
            end_seconds = start_seconds + song["duration_seconds"]
            id_to_hash_str = f"{name}\n{index}\n{start_seconds}\n{end_seconds}".encode(
                "utf-8"
            )
            hash_obj = sha224()
            hash_obj.update(id_to_hash_str)
            hash_bytes = hash_obj.digest()
            _id = np.frombuffer(hash_bytes[:8], dtype=np.int64)[0]  # type: ignore
            current_chunk_data = []
            embedding_name = name
            for chunk_embedding, chunk_metadata in zip(
                song["chunk_embeddings"], song["chunk_metadata"]
            ):
                chunk_index = chunk_metadata["index"]
                chunk_start_seconds = chunk_metadata["start_seconds"]
                chunk_end_seconds = chunk_metadata["end_seconds"]
                chunk_hash_key = f"{name}\n{chunk_index}\n{chunk_start_seconds}\n{chunk_end_seconds}".encode(
                    "utf-8"
                )
                hash_obj = sha224()
                hash_obj.update(chunk_hash_key)
                hash_bytes = hash_obj.digest()
                chunk_id = np.frombuffer(hash_bytes[:8], dtype=np.int64)[0]
                current_chunk_data.append(
                    ChunkedEmbedding(
                        id=chunk_id,
                        parent_id=embedding_name,
                        start_seconds=chunk_start_seconds,
                        end_seconds=chunk_end_seconds,
                        embedding=chunk_embedding,
                    )
                )

            song_embed = SongEmbedding(
                name=name,
                embedding=song["centroid"],
                window=window_seconds,
                hop=hop_seconds,
                sample_rate=sample_rate,
                offset=start_seconds,
                chunk_ids=[i.id for i in current_chunk_data],
                track_id=str(_id),
            )
            chunk_data.extend(current_chunk_data)
            song_data.append(song_embed)

        return song_data, chunk_data

    def add_embedding_to_db(self, file_name, embedding: dict, settings: UploadSettings):
        self.logger.info(
            "Uploading embedding for %s to Milvus", embedding.get("music_file")
        )
        self.logger.debug("Received upload settings: %s", asdict(settings))
        self.milvus_client.load_collection("embedding")
        songs, chunk_embeddings = self.load_from_json(embedding)
        duplicates = self.feature_pipeline.scan_for_dups(songs, settings)
        songs_payload = list(map(asdict, songs))
        chunk_payload = list(map(asdict, chunk_embeddings))
        for item in songs_payload:
            item.pop("track_id", None)
        new_name = self.feature_pipeline.rename(
            file_name, settings, music_file=embedding.get("music_file")
        )
        if len(duplicates) == len(songs_payload):
            self.logger.debug("Not upserting song embeddings because of deduplication")
        else:
            try:
                if songs_payload:
                    self.milvus_client.upsert("embedding", songs_payload)
                if chunk_payload:
                    self.milvus_client.upsert("chunked_embedding", chunk_payload)
                if songs_payload:
                    self.milvus_client.flush("embedding")
                if chunk_payload:
                    self.milvus_client.flush("chunked_embedding")
            except Exception:
                self.logger.exception("Failed to upsert embeddings to Milvus")
        self.logger.debug(
            "Prepared embedding payload for Milvus. Songs=%d, chunks=%d, duplicates=%d",
            len(songs),
            len(chunk_embeddings),
            len(duplicates),
        )
        all_duplicates = bool(songs_payload) and len(duplicates) >= len(songs_payload)
        return {
            "duplicates": duplicates,
            "renamedFile": new_name,
            "allDuplicates": all_duplicates,
        }

    def handle_connection(self, conn: socket.socket) -> None:
        with conn:
            reader = conn.makefile("r", encoding="utf-8")
            writer = conn.makefile("w", encoding="utf-8")
            try:
                line = reader.readline()
                if not line:
                    self.logger.debug("Received empty request; closing connection")
                    return

                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    self.logger.error("Invalid JSON payload: %s", exc)
                    self._write_response(
                        writer, {"status": "error", "message": f"invalid json: {exc}"}
                    )
                    return

                music_file = payload.get("music_file")
                music_name = payload.get("name")
                settings = UploadSettings.from_payload(payload.get("settings"))
                cue_file = payload.get("cue_file") or None
                self.logger.debug("Payload: %s", payload)
                self.logger.debug("Normalized upload settings: %s", asdict(settings))
                self.logger.debug(
                    "Received embedding request for %s (cue=%s)", music_file, cue_file
                )

                if not music_file:
                    self.logger.error("Request missing music_file field")
                    self._write_response(
                        writer,
                        {"status": "error", "message": "music_file is required"},
                    )
                    return

                summary: Optional[dict] = None
                try:
                    result = self.model.embed_music(music_file, music_name, cue_file)
                    summary = self.add_embedding_to_db(music_name, result, settings)
                except Exception as exc:  # pragma: no cover - propagate to client
                    self.logger.exception("Embedding failed for %s", music_file)
                    self._write_response(
                        writer, {"status": "error", "message": str(exc)}
                    )
                    return

                response_payload = {"status": "ok"}
                if isinstance(summary, dict):
                    response_payload.update(summary)
                self._write_response(writer, response_payload)
            finally:
                self.logger.debug("Closing connection")
                writer.close()
                reader.close()

    def _write_response(self, writer, payload) -> None:
        self.logger.debug("Sending response: %s", payload)
        writer.write(json.dumps(payload))
        writer.write("\n")
        writer.flush()


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)
    server = EmbedSocketServer()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.getLogger("navidrome.embed_server").info(
            "Embedding server shutting down"
        )


if __name__ == "__main__":
    main()
