import os
import gc
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import librosa
import torch
import logging
from transformers.audio_utils import AudioInput, make_list_of_audio
from transformers.processing_utils import Unpack
from transformers import Cache, MusicFlamingoForConditionalGeneration, MusicFlamingoProcessor, MusicFlamingoForConditionalGeneration
from transformers.generation import LogitsProcessor, LogitsProcessorList
from transformers.models.musicflamingo.processing_musicflamingo import MusicFlamingoProcessorKwargs
from transformers.utils.quantization_config import FineGrainedFP8Config
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from sentence_transformers import SentenceTransformer

from enrichment import enrich_and_concatenate

DESCRIBE_PROMPT = "Describe this track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, lyrical themes, and the overall mood it creates."
LYRICS_CHECK_PROMPT = "Does this piece have lyrics? Answer with Yes or No."
LYRICS_PROMPT = (
    "Transcribe the lyrics from this track in a structured lyric sheet. "
    "If there are sections without vocals, skip them and only return the words that are sung or spoken. "
    "If there are repeating lyrics, just note how many times with: [repeats <n> times]."
)
NO_LYRICS_RESPONSE = "This song contains no lyrics"
LONG_SONG_SECONDS = 18 * 60
AUDIO_SAMPLE_RATE = 16000
MAX_NEW_TOKENS = 2048


@dataclass
class GenerationSettings:
    sample_method: str = "greedy"  # greedy, top_p, top_k, beam, temperature, contrastive
    do_sample: Optional[bool] = True
    max_new_tokens: int = MAX_NEW_TOKENS
    min_new_tokens: int = 0
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    typical_p: float = 1.0
    length_penalty: float = 1.0
    num_beams: int = 1
    num_return_sequences: int = 1
    early_stopping: bool = True
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    dry_multiplier: float = 0.3
    dry_base: float = 1.75
    dry_allowed_length: int = 5
    dry_penalty_last_n: Optional[int] = None
    dry_sequence_breakers: Sequence[str] = field(default_factory=lambda: ("\n", ":", "\"", "*"))
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


class DryLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        dry_multiplier: float,
        dry_base: float,
        dry_allowed_length: int,
        dry_penalty_last_n: Optional[int] = None,
        dry_sequence_breakers: Optional[Sequence[str]] = None,
    ):
        if dry_multiplier < 0:
            raise ValueError("dry_multiplier must be >= 0")
        if dry_base <= 1.0:
            raise ValueError("dry_base must be > 1")
        if dry_allowed_length < 0:
            raise ValueError("dry_allowed_length must be >= 0")
        self.tokenizer = tokenizer
        self.dry_multiplier = dry_multiplier
        self.dry_base = dry_base
        self.dry_allowed_length = dry_allowed_length
        self.dry_penalty_last_n = dry_penalty_last_n
        breakers = dry_sequence_breakers or ()
        breaker_ids: set[int] = set()
        for item in breakers:
            token_ids = tokenizer.encode(item, add_special_tokens=False)
            breaker_ids.update(token_ids)
        self.breaker_ids = breaker_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.dry_multiplier <= 0 or self.dry_allowed_length <= 0:
            return scores

        batch_size, _ = input_ids.shape
        for batch_idx in range(batch_size):
            seq = input_ids[batch_idx].tolist()
            if len(seq) <= self.dry_allowed_length:
                continue

            lookback_start = 0
            if self.dry_penalty_last_n is not None and self.dry_penalty_last_n > 0:
                lookback_start = max(0, len(seq) - self.dry_penalty_last_n)

            last_breaker = -1
            if self.breaker_ids:
                for i in range(len(seq) - 1, lookback_start - 1, -1):
                    if seq[i] in self.breaker_ids:
                        last_breaker = i
                        break

            suffix_start = max(last_breaker + 1, lookback_start)
            suffix = seq[suffix_start:]
            if not suffix:
                continue

            dist_since_breaker: List[int] = [0] * len(seq)
            run = 0
            for idx, token_id in enumerate(seq):
                if token_id in self.breaker_ids:
                    run = 0
                else:
                    run += 1
                dist_since_breaker[idx] = run

            max_match: Dict[int, int] = {}
            suffix_len = len(suffix)
            for i in range(lookback_start + 1, len(seq)):
                token_id = seq[i]
                if token_id in self.breaker_ids:
                    continue
                prev_index = i - 1
                if prev_index < 0:
                    continue
                max_k = min(
                    suffix_len,
                    dist_since_breaker[prev_index],
                    prev_index - lookback_start + 1,
                )
                if max_k <= 0:
                    continue
                k = 0
                while k < max_k and seq[prev_index - k] == suffix[-1 - k]:
                    k += 1
                if k >= self.dry_allowed_length:
                    current = max_match.get(token_id, 0)
                    if k > current:
                        max_match[token_id] = k

            for token_id, match_len in max_match.items():
                penalty = self.dry_multiplier * (self.dry_base ** (match_len - self.dry_allowed_length))
                scores[batch_idx, token_id] -= penalty

        return scores


class TimeToFirstTokenProcessor(LogitsProcessor):
    def __init__(self, start_time: float):
        self.start_time = start_time
        self.first_token_time: Optional[float] = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        return scores


class CustomMusicFlamingo(MusicFlamingoForConditionalGeneration):
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Union[Cache, None] = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        audio_times: torch.Tensor | None = None,
        cache_position: torch.LongTensor | None = None,
        is_first_iteration: bool | None = False,
        **kwargs,
    ):
        return super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            input_features=input_features,
            input_features_mask=input_features_mask,
            audio_times=audio_times,
            cache_position=cache_position,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

@contextmanager
def suppress_logs(level=logging.CRITICAL):
    old = logging.root.manager.disable
    logging.disable(level)   # disables all logs <= level
    try:
        yield
    finally:
        logging.disable(old)
        

@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield

@contextmanager
def suppress_generate_device_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You are calling .generate\\(\\) with the `input_ids` being on a device type different than your model's device.*",
            category=UserWarning,
        )
        yield

class MusicFlamingoAudioProcessor(MusicFlamingoProcessor):
    def __init__(
        self,
        feature_extractor,
        tokenizer,
        chat_template=None,
        audio_token="<sound>",
        sound_bos_token="<|sound_bos|>",
        sound_eos_token="<|sound_eos|>",
    ):
        super().__init__(feature_extractor, tokenizer, chat_template, audio_token, sound_bos_token, sound_eos_token)

    def __call__(self, # type: ignore
        text: Optional[Union[str, List[str]]],
        audio: AudioInput,
        output_labels: Optional[bool] = False,
        **kwargs: Unpack[MusicFlamingoProcessorKwargs]):       
        if text is None:
            input_audio_count = len(make_list_of_audio(audio))
            text = [""] * input_audio_count
        return super().__call__(text, audio, output_labels, **kwargs)


class MusicFlamingo:
    def __init__(
        self,
        path,
        device="cuda",
        audio_device: Optional[str] = None,
        llm_device: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        dequantize_fp8: bool = False,
        audio_only: bool = False,
        offload_audio_tower: Optional[bool] = None,
        reuse_audio_embeds: bool = True,
        generation_settings: Optional[GenerationSettings] = None,
        log_tps: bool = True,
        log_lyrics_check: bool = True,
        max_audio_seconds: Optional[float] = None,
    ):
        self.path = path
        self.device_str = device
        self.audio_device_str = audio_device or device
        self.llm_device_str = llm_device or ("cpu" if audio_only else device)
        self.audio_device = torch.device(self.audio_device_str)
        self.llm_device = torch.device(self.llm_device_str)
        self.device = self.audio_device
        self.audio_only = audio_only
        self.reuse_audio_embeds = reuse_audio_embeds
        if offload_audio_tower is None:
            offload_audio_tower = not audio_only
        self.offload_audio_tower = offload_audio_tower
        device_map = self._build_device_map()
        quantization_config = None
        if dequantize_fp8:
            quantization_config = FineGrainedFP8Config(
                activation_scheme="dynamic",
                weight_block_size=(128, 128),
                dequantize=True,
                modules_to_not_convert=["audio_tower", "multi_modal_projector", "lm_head"],
            )
        self.music_flamingo = MusicFlamingo.load_music_flamingo(
            self.path,
            device_map=device_map,
            attn_implementation=attn_implementation,
            quantization_config=quantization_config,
        )
        self.music_processor = MusicFlamingo.load_music_processor(self.path)
        #self.llm = self.load_flamingo_llm()
        self.describe_music_prompt = DESCRIBE_PROMPT
        self.lyrics_prompt = LYRICS_PROMPT
        self.lyrics_check_prompt = LYRICS_CHECK_PROMPT
        self.generation_settings = generation_settings or GenerationSettings()
        self.log_tps = log_tps
        self.log_lyrics_check = log_lyrics_check
        self.max_audio_seconds = max_audio_seconds
        if self.generation_settings.eos_token_id is None:
            self.generation_settings.eos_token_id = self.music_processor.tokenizer.eos_token_id
        if self.generation_settings.pad_token_id is None:
            self.generation_settings.pad_token_id = self.music_processor.tokenizer.pad_token_id
        self.max_new_tokens = self.generation_settings.max_new_tokens
        
    def _build_device_map(self) -> Dict[str, str]:
        audio_device = str(self.audio_device)
        llm_device = "cpu" if self.audio_only else str(self.llm_device)
        return {
            "audio_tower": audio_device,
            "multi_modal_projector": audio_device,
            "language_model": llm_device,
        }

    def unload_all(self):
        has_meta = any(getattr(param, "is_meta", False) for param in self.music_flamingo.parameters())
        if not has_meta:
            self.music_flamingo.cpu()

    def will_cause_oom(self, audio: Union[str, torch.Tensor]) -> bool:
        duration = self._get_audio_duration(audio)
        return duration >= LONG_SONG_SECONDS

    def _get_audio_duration(self, audio: Union[str, torch.Tensor]) -> float:
        if isinstance(audio, str):
            with suppress_logs(), suppress_output():
                return float(librosa.get_duration(path=audio))
        if isinstance(audio, torch.Tensor):
            samples = audio.shape[-1]
            return float(samples) / float(AUDIO_SAMPLE_RATE)
        raise TypeError("audio must be a file path or torch.Tensor")

    def prepare_audio_context(self, audio: Union[str, torch.Tensor], *, compute_audio_embeds: bool = True):
        if isinstance(audio, str):
            audio = self.load_audio(audio, max_duration_seconds=self.max_audio_seconds)
        dummy_text = self.music_processor.audio_token
        inputs = self.music_processor(
            text=dummy_text,
            audio=audio,
            return_tensors="pt",
        )  # type: ignore
        audio_token_count = int((inputs["input_ids"] == self.music_flamingo.config.audio_token_id).sum().item())
        audio_inputs = {
            "input_features": inputs["input_features"],
            "input_features_mask": inputs["input_features_mask"],
        }
        if "audio_times" in inputs:
            audio_inputs["audio_times"] = inputs["audio_times"]
        audio_inputs = {k: v.to(self.audio_device) for k, v in audio_inputs.items()}
        audio_embeds = None
        if compute_audio_embeds:
            self._move_audio_modules(self.audio_device)
            with torch.inference_mode():
                audio_embeds = self.music_flamingo.get_audio_features(**audio_inputs)
            if self.offload_audio_tower:
                self._move_audio_modules("cpu")

        return {
            "audio_inputs": audio_inputs,
            "audio_token_count": audio_token_count,
            "audio_embeds": audio_embeds,
        }

    def extract_embedding(
        self,
        audio: Union[str, torch.Tensor],
        existing_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if existing_embedding is not None:
            return existing_embedding
        audio_context = self.prepare_audio_context(audio)
        embedding = audio_context["audio_embeds"].to("cpu").detach()
        return embedding

    def generate_description(self, audio_context, generation_overrides: Optional[Dict[str, Any]] = None):
        return self._generate_from_prompt(self.describe_music_prompt, audio_context, generation_overrides)

    def generate_lyrics(self, audio_context, generation_overrides: Optional[Dict[str, Any]] = None):
        if not self.has_lyrics(audio_context):
            return NO_LYRICS_RESPONSE
        return self._generate_from_prompt(self.lyrics_prompt, audio_context, generation_overrides)

    def has_lyrics(self, audio_context) -> bool:
        response = self._generate_from_prompt(
            self.lyrics_check_prompt,
            audio_context,
            {"max_new_tokens": 8},
        )
        parsed = self._parse_yes_no(response)
        if self.log_lyrics_check:
            normalized = response.strip().lower()
            first_token = normalized.split()[0] if normalized else ""
            print(
                "Lyrics check response:"
                f" raw={response!r} normalized={normalized!r} first_token={first_token!r} parsed={parsed}"
            )
        return parsed

    def describe_with_embedding_and_lyrics(self, audio: Union[str, torch.Tensor]):
        audio_context = self.prepare_audio_context(audio)
        description = ""
        lyrics = ""
        if not self.audio_only:
            description = self.generate_description(audio_context)
            lyrics = self.generate_lyrics(audio_context)
        embedding = audio_context["audio_embeds"].to("cpu").detach()
        if self.offload_audio_tower:
            self._move_audio_modules(torch.device("cpu"))
        return embedding, description, lyrics

    def describe_text_only(self, audio: Union[str, torch.Tensor]):
        if self.audio_only:
            raise RuntimeError("Text generation is disabled for this MusicFlamingo instance.")
        audio_context = self.prepare_audio_context(audio, compute_audio_embeds=self.reuse_audio_embeds)
        description = self.generate_description(audio_context)
        lyrics = self.generate_lyrics(audio_context)
        if self.offload_audio_tower:
            self._move_audio_modules(torch.device("cpu"))
        return description, lyrics

    def _generate_from_prompt(self, prompt: str, audio_context, generation_overrides: Optional[Dict[str, Any]] = None):
        text = self.prepare_model_input(prompt_text=prompt, audio_token_count=audio_context["audio_token_count"])
        text_inputs = self.music_processor.tokenizer(text, return_tensors="pt", padding=True)
        audio_embeds = audio_context.get("audio_embeds")
        use_audio_embeds = self.reuse_audio_embeds and audio_embeds is not None
        model_inputs: Dict[str, Any]
        if use_audio_embeds:
            input_ids = text_inputs["input_ids"].to(self.llm_device)
            inputs_embeds = self.music_flamingo.get_input_embeddings()(input_ids)
            audio_token_mask = (input_ids == self.music_flamingo.config.audio_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask,
                audio_embeds.to(self.llm_device),
            )
            model_inputs = {
                "input_ids": input_ids,
                "inputs_embeds": inputs_embeds,
            }
            if "attention_mask" in text_inputs:
                model_inputs["attention_mask"] = text_inputs["attention_mask"].to(self.llm_device)
        else:
            self._move_audio_modules(self.audio_device)
            model_inputs = {**text_inputs, **audio_context["audio_inputs"]}
            model_inputs = {
                key: value.to(self.llm_device) if isinstance(value, torch.Tensor) else value
                for key, value in model_inputs.items()
            }
        generation_kwargs = self._build_generation_kwargs(generation_overrides)
        total_input_tokens = int(text_inputs["input_ids"].numel())
        start_time = time.perf_counter()
        ttfb_processor: Optional[TimeToFirstTokenProcessor] = None
        if self.log_tps:
            ttfb_processor = TimeToFirstTokenProcessor(start_time)
            existing_processors = generation_kwargs.get("logits_processor")
            if existing_processors is None:
                generation_kwargs["logits_processor"] = LogitsProcessorList([ttfb_processor])
            else:
                if isinstance(existing_processors, LogitsProcessor):
                    existing_processors = LogitsProcessorList([existing_processors])
                elif not isinstance(existing_processors, LogitsProcessorList):
                    existing_processors = LogitsProcessorList(list(existing_processors))
                existing_processors.append(ttfb_processor)
                generation_kwargs["logits_processor"] = existing_processors
        with torch.inference_mode():
            with suppress_generate_device_warning():
                outputs = self.music_flamingo.generate(
                    **model_inputs,
                    **generation_kwargs,
                )
        elapsed = time.perf_counter() - start_time
        generated = outputs[:, text_inputs["input_ids"].shape[1]:]
        if self.log_tps and elapsed > 0:
            first_token_time = ttfb_processor.first_token_time if ttfb_processor else None
            if first_token_time is not None:
                prefill_time = max(first_token_time - start_time, 1e-6)
                gen_time = max(elapsed - prefill_time, 1e-6)
                gen_tokens = int(generated.numel())
                prefill_tps = total_input_tokens / prefill_time if prefill_time > 0 else 0.0
                gen_tps = gen_tokens / gen_time if gen_time > 0 else 0.0
                print(f"Prefill tokens/s: {prefill_tps:.2f}", flush=True)
                print(f"Generation tokens/s: {gen_tps:.2f}", flush=True)
            else:
                tps = total_input_tokens / elapsed
                print(f"Tokens/s: {tps:.2f}", flush=True)
        decoded_outputs = self.music_processor.batch_decode(
            generated, skip_special_tokens=True
        )
        return decoded_outputs[0]

    def _move_audio_modules(self, device: torch.device) -> None:
        gc.collect()
        torch.cuda.empty_cache()
        for name in ("audio_tower", "multi_modal_projector"):
            module = getattr(self.music_flamingo, name, None)
            if module is None:
                continue
            module.to(device)

    @staticmethod
    def _parse_yes_no(text: str) -> bool:
        normalized = text.strip().lower()
        if not normalized:
            return False
        first = normalized.split()[0].strip(".,:;!?\"'")
        if first in ("yes", "y"):
            return True
        if first in ("no", "n"):
            return False
        if "yes" in normalized:
            return True
        if "no" in normalized:
            return False
        return False
        
    @staticmethod
    def load_music_flamingo(path, **kwargs):
        with suppress_logs(), suppress_output():
            return CustomMusicFlamingo.from_pretrained(path, **kwargs).eval()
    
    @staticmethod
    def load_music_processor(path):
        return MusicFlamingoAudioProcessor.from_pretrained(path)
    
    @staticmethod
    def load_audio(path, max_duration_seconds: Optional[float] = None):
        with suppress_logs(), suppress_output():
            if max_duration_seconds is not None and max_duration_seconds > 0:
                audio, _ = librosa.load(path, sr=AUDIO_SAMPLE_RATE, duration=max_duration_seconds)
            else:
                audio, _ = librosa.load(path, sr=AUDIO_SAMPLE_RATE)
        return audio

    
    
    def prepare_model_input(self, prompt_text: Optional[str] = None, audio_token_count: int | None = None):
        if prompt_text is None:
            prompt_text = self.describe_music_prompt
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "audio", "path":  "<sound>"},
                ],
            }
        ]
        text_prompt = self.music_processor.apply_chat_template(
            conversation,
            tokenize=False, # type: ignore
            add_generation_prompt=True, # type: ignore
        )

        if audio_token_count is not None:
            if audio_token_count < 1:
                raise ValueError("audio_token_count must be >= 1")
            expanded_audio = (
                self.music_processor.sound_bos_token
                + (self.music_processor.audio_token * audio_token_count)
                + self.music_processor.sound_eos_token
            )
            text_prompt = text_prompt.replace(self.music_processor.audio_token, expanded_audio)

        return text_prompt

    def update_generation_settings(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if key == "extra_kwargs":
                if not isinstance(value, dict):
                    raise ValueError("extra_kwargs must be a dict")
                self.generation_settings.extra_kwargs.update(value)
                continue
            if key == "dry":
                key = "dry_multiplier"
            if not hasattr(self.generation_settings, key):
                raise ValueError(f"Unknown generation setting: {key}")
            setattr(self.generation_settings, key, value)
            if key == "max_new_tokens":
                self.max_new_tokens = value

    def _build_generation_kwargs(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        settings = self.generation_settings
        override_values: Dict[str, Any] = dict(overrides or {})
        settings_values = {key: getattr(settings, key) for key in settings.__dataclass_fields__}
        extra_kwargs = settings_values.pop("extra_kwargs", {})
        if "extra_kwargs" in override_values:
            extra_override = override_values.pop("extra_kwargs")
            if not isinstance(extra_override, dict):
                raise ValueError("extra_kwargs override must be a dict")
            extra_kwargs = {**extra_kwargs, **extra_override}
        for key in list(override_values.keys()):
            if key in settings_values:
                settings_values[key] = override_values.pop(key)

        method = (settings_values["sample_method"] or "greedy").lower()
        method = method.replace("-", "_")
        if method in ("greedy", "deterministic"):
            method_do_sample = False
        elif method in ("beam", "beam_search"):
            method_do_sample = False
        elif method in ("top_p", "nucleus"):
            method_do_sample = True
        elif method in ("top_k",):
            method_do_sample = True
        elif method in ("temperature", "sampling"):
            method_do_sample = True
        elif method in ("contrastive",):
            method_do_sample = False
        else:
            raise ValueError(f"Unknown sample_method: {settings_values['sample_method']}")

        do_sample = settings_values["do_sample"] if settings_values["do_sample"] is not None else method_do_sample

        generation_kwargs: Dict[str, Any] = {
            "do_sample": do_sample,
            "max_new_tokens": settings_values["max_new_tokens"],
            "min_new_tokens": settings_values["min_new_tokens"],
            "temperature": settings_values["temperature"],
            "top_p": settings_values["top_p"],
            "top_k": settings_values["top_k"],
            "typical_p": settings_values["typical_p"],
            "length_penalty": settings_values["length_penalty"],
            "num_beams": settings_values["num_beams"],
            "num_return_sequences": settings_values["num_return_sequences"],
            "early_stopping": settings_values["early_stopping"],
            "eos_token_id": settings_values["eos_token_id"],
            "pad_token_id": settings_values["pad_token_id"],
        }

        if method in ("beam", "beam_search"):
            generation_kwargs["num_beams"] = max(4, settings_values["num_beams"])
            generation_kwargs["do_sample"] = False
        elif method in ("contrastive",):
            generation_kwargs["do_sample"] = False

        if generation_kwargs.get("num_beams", 1) <= 1:
            generation_kwargs.pop("early_stopping", None)

        if generation_kwargs["temperature"] is not None and generation_kwargs["temperature"] <= 0:
            raise ValueError("temperature must be > 0")
        if override_values:
            generation_kwargs.update(override_values)
        if extra_kwargs:
            generation_kwargs.update(extra_kwargs)

        dry_multiplier = settings_values.get("dry_multiplier", 0.0)
        if dry_multiplier and dry_multiplier > 0:
            dry_processor = DryLogitsProcessor(
                tokenizer=self.music_processor.tokenizer,
                dry_multiplier=dry_multiplier,
                dry_base=settings_values["dry_base"],
                dry_allowed_length=settings_values["dry_allowed_length"],
                dry_penalty_last_n=settings_values["dry_penalty_last_n"],
                dry_sequence_breakers=settings_values["dry_sequence_breakers"],
            )
            existing_processors = generation_kwargs.get("logits_processor")
            if existing_processors is None:
                generation_kwargs["logits_processor"] = LogitsProcessorList([dry_processor])
            else:
                if isinstance(existing_processors, LogitsProcessor):
                    existing_processors = LogitsProcessorList([existing_processors])
                elif not isinstance(existing_processors, LogitsProcessorList):
                    existing_processors = LogitsProcessorList(list(existing_processors))
                existing_processors.append(dry_processor)
                generation_kwargs["logits_processor"] = existing_processors

        if not generation_kwargs.get("do_sample", False):
            generation_kwargs.pop("temperature", None)
            generation_kwargs.pop("top_p", None)
            generation_kwargs.pop("top_k", None)
            generation_kwargs.pop("typical_p", None)

        return generation_kwargs
    
    @staticmethod
    def flatten_and_enrich_embedding(embedding: torch.FloatTensor):
        embedding = embedding.to("cuda").to(torch.float32)
        return enrich_and_concatenate(embedding)
        
    
    
    
class QwenEmbedder:
    def __init__(
        self,
        batch_size: int = 64,
        device: Optional[str] = None,
        show_progress_bar: bool = False,
    ):
        self._model = None
        self.batch_size = batch_size
        self.device = device
        self.show_progress_bar = show_progress_bar

    def _load_model(self):
        kwargs: Dict[str, Any] = {}
        if self.device:
            kwargs["device"] = self.device
        self._model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", **kwargs)

    def encode_documents(self, strings: List[str]):
        if not strings:
            return []
        if self._model is None:
            self._load_model()
        assert self._model is not None
        return self._model.encode(
            strings,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            normalize_embeddings=True,
        )
        
        
def test():
    songs = ["/home/henry/projects/navidrome/music/Lorde-Pure_Heroine-24BIT-WEB-FLAC-2013-TVRf/04-lorde-ribs.flac",
        "/home/henry/projects/navidrome/music/Lorde-Pure_Heroine-24BIT-WEB-FLAC-2013-TVRf/01-lorde-tennis_court.flac"]
    mf = MusicFlamingo("./music_flamingo_fp8")
    print("First Song:")
    embedding, description, lyrics = mf.describe_with_embedding_and_lyrics(songs[0])
    print(f"Embedding shape: {tuple(embedding.shape)}")
    print("Description:")
    print(description)
    print("Lyrics:")
    print(lyrics)
    print("Second Song:")
    embedding1, description1, lyrics1 = mf.describe_with_embedding_and_lyrics(songs[1])
    print(f"Embedding shape: {tuple(embedding1.shape)}")
    print("Description:")
    print(description1)
    print("Lyrics:")
    print(lyrics1)
    mf.unload_all()
    del mf
    gc.collect()
    qwen = QwenEmbedder()
    qwen_embedding = qwen.encode_documents([description, description1, lyrics, lyrics1])
    print(f"Qwen embedding shape: {qwen_embedding.shape}")
    



if __name__ == "__main__":
    test()
