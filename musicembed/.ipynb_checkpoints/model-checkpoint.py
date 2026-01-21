import json
import os
import sys

import librosa
import torch
from transformers import AutoModel, AutoProcessor, MusicFlamingoPreTrainedModel, MusicFlamingoForConditionalGeneration
from transformers.models.musicflamingo.modeling_musicflamingo import MusicFlamingoMultiModalProjector


class NoOpCall:
    def __init__(self):
        pass
    
    def __call__(**kwargs):
        return kwargs

class MusicFlamingoPreProcessorModel(MusicFlamingoForConditionalGeneration):
    def __init__(self, config):
        super(MusicFlamingoForConditionalGeneration, self).__init__(config)
        print(config)
        self.vocab_size = config.text_config.vocab_size
        self.audio_tower = AutoModel.from_config(config.audio_config)
        self.multi_modal_projector = MusicFlamingoMultiModalProjector(config)
        self.language_model = NoOpCall()
        self.post_init()
    
    
    

SGLANG_PYTHON_PATH = os.path.join(os.path.dirname(__file__), "sglang", "python")
if SGLANG_PYTHON_PATH not in sys.path:
    sys.path.insert(0, SGLANG_PYTHON_PATH)

_sglang_runner = None
_sglang_runner_path = None


def _get_sglang_runner(model_path, attention_backend="triton", tp_size=1):
    global _sglang_runner, _sglang_runner_path
    if _sglang_runner is not None and _sglang_runner_path == model_path:
        return _sglang_runner

    from sglang.bench_one_batch import load_model
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.server_args import PortArgs, ServerArgs

    model_override = {"architectures": ["MusicFlamingoQwen2ForCausalLM"], "model_type": "qwen2"}
    server_args_kwargs = {
        "model_path": model_path,
        "tokenizer_path": model_path,
        "trust_remote_code": True,
        "tp_size": tp_size,
        "disable_radix_cache": True,
        "json_model_override_args": json.dumps(model_override),
        "fp8_gemm_runner_backend": "cutlass",
        "disable_cuda_graph": True,
        "sampling_backend": "pytorch",
        "grammar_backend": "none",
    }
    if attention_backend:
        server_args_kwargs["attention_backend"] = attention_backend

    server_args = ServerArgs(**server_args_kwargs)
    _set_envs_and_config(server_args)
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)

    port_args = PortArgs.init_new(server_args)
    model_runner, _ = load_model(server_args, port_args, gpu_id=0, tp_rank=0)
    _sglang_runner = model_runner
    _sglang_runner_path = model_path
    return _sglang_runner

def load_music_flamingo_processor(path):
    return AutoProcessor.from_pretrained(path)

def prepare_music_embedding(model, processor, music_path):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path":  music_path},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=text_prompt, audio=music_path, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        audio_embeds = model.get_audio_features(
            inputs["input_features"],
            inputs["input_features_mask"],
            audio_times=inputs.get("audio_times"),
        )
        inputs_embeds = model.get_input_embeddings()(inputs["input_ids"])
        audio_token_mask = (inputs["input_ids"] == model.config.audio_token_id).unsqueeze(-1)
        inputs_embeds = inputs_embeds.masked_scatter(audio_token_mask, audio_embeds)
    
    return audio_embeds, audio_token_mask, input_embeds

def load_music_flamingo_model(path):
    processor = load_music_flamingo_processor(path)
    model = _get_sglang_runner(path)
    
    
def load_qwen_embedder(path):
    # dont write this yet
    pass

def inference_music_flamingo(model, music_path, prompt, embedding=True): 
    # inferences music flamingo with a default prompt
    processor = getattr(model, "processor", None)
    if processor is None:
        raise ValueError("model.processor is missing. Use load_music_flamingo to load the model.")

    if isinstance(music_path, str):
        audio, _ = librosa.load(music_path, sr=16000)
    else:
        audio = music_path

    
   

    from sglang.bench_one_batch import decode, extend
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    model_path = getattr(model, "name_or_path", None) or getattr(model.config, "_name_or_path", "./music_flamingo_fp8")
    runner = _get_sglang_runner(model_path)
    input_embeds = inputs_embeds.squeeze(0).detach().cpu().float().tolist()
    sampling_params = SamplingParams(temperature=0.0, max_new_tokens=256)
    fake_ids = [1] * len(input_embeds)
    req = Req(
        rid="0",
        origin_input_text="",
        origin_input_ids=fake_ids,
        sampling_params=sampling_params,
        input_embeds=input_embeds,
    )
    req.fill_ids = req.origin_input_ids
    req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
    req.logprob_start_len = len(req.origin_input_ids) - 1

    next_token_ids, _, batch_state = extend([req], runner)
    output_ids = []
    ids = next_token_ids.detach().cpu().tolist()
    output_ids.append(int(ids[0]))
    for _ in range(sampling_params.max_new_tokens - 1):
        next_token_ids, _ = decode(next_token_ids, batch_state, runner)
        ids = next_token_ids.detach().cpu().tolist()
        output_ids.append(int(ids[0]))

    decoded = processor.tokenizer.decode(output_ids, skip_special_tokens=True)
    audio_embeds = audio_embeds.detach().cpu()

    return audio_embeds, decoded

def inference_music_flamingo_lyrics(model, music_path, has_lyric_prompt, lyric_extract_prompt):
    # dont write this yet
    pass

def inference_qwen(model, text):
    # dont write this yet
    pass
