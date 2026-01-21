import json
import os
import sys
from typing import List, Optional, Union

import librosa
import torch
import logging
from transformers.audio_utils import AudioInput, make_list_of_audio
from transformers.processing_utils import Unpack
from transformers import AutoModel, MusicFlamingoForConditionalGeneration, MusicFlamingoProcessor, MusicFlamingoForConditionalGeneration
from transformers.models.musicflamingo.modeling_musicflamingo import MusicFlamingoMultiModalProjector
from transformers.models.musicflamingo.processing_musicflamingo import MusicFlamingoProcessorKwargs
from contextlib import contextmanager, redirect_stdout, redirect_stderr

from sglang.bench_one_batch import decode, extend
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.bench_one_batch import load_model
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.server_args import PortArgs, ServerArgs

DESCRIBE_PROMPT = "Describe this track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, and overall mood it creates."



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

class NoOpCall:
    def __init__(self):
        pass
    
    def __call__(**kwargs):
        return kwargs

class MusicFlamingoPreProcessorModel(MusicFlamingoForConditionalGeneration):
    def __init__(self, config):
        super(MusicFlamingoForConditionalGeneration, self).__init__(config)
        self.vocab_size = config.text_config.vocab_size
        self.audio_tower = AutoModel.from_config(config.audio_config)
        self.multi_modal_projector = MusicFlamingoMultiModalProjector(config)
        self.language_model = NoOpCall()
        self.post_init()




class MockTokenizer:
    def __init__(self) -> None:
        self.init_kwargs = {}
    
    def __call__(self, text, **kwds):
        return {}
    
        

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

    def __call__(self,
        text: Optional[Union[str, List[str]]],
        audio: AudioInput,
        output_labels: Optional[bool] = False,
        **kwargs: Unpack[MusicFlamingoProcessorKwargs]):       
        if text is None:
            input_audio_count = len(make_list_of_audio(audio))
            text = [""] * input_audio_count
        return super().__call__(text, audio, output_labels, **kwargs)






class MusicFlamingo:
    def __init__(self, path, device="cuda"):
        self.path = path
        self.device = torch.device(device)
        self.music_embedder = MusicFlamingo.load_audio_embedder(self.path, device_map=self.device)
        self.music_processor = MusicFlamingo.load_music_processor(self.path)
        self.llm = self.load_flamingo_llm()
        self.describe_music_prompt = DESCRIBE_PROMPT
        
    def embed_music_from_path(self, path):
        loaded = self.load_audio(path)
        prepared = self.prepare_music(loaded)
        embedded = self.embed_music(prepared)
        return embedded.to("cpu").detach()
    
    def inference_llm(self, audio_embedding):
        text = self.prepare_model_input()
        inputs = self.music_processor(
                    text=text,
                    audio=None,
                    return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        attention_mask = inputs["attention_mask"]
        input_features = audio_embedding["input_features"]
        input_features_mask = audio_embedding["input_features_mask"]
        audio_times = audio_embedding["audio_times"]
        self.
        

    def prepare_music(self, audio):
        processed = self.music_processor(None, audio, return_tensors="pt").to(self.device) # type: ignore
        del processed["input_ids"]
        del processed["attention_mask"]
        return processed
    
    def embed_music(self, inputs):
        with torch.inference_mode():
            embeddings = self.music_embedder.get_audio_features(**inputs)
        return embeddings
        
    @staticmethod
    def load_music_flamingo(path, **kwargs):
        with suppress_logs(), suppress_output():
            return MusicFlamingoForConditionalGeneration.from_pretrained(path, **kwargs).eval()
    
    @staticmethod
    def load_music_processor(path):
        return MusicFlamingoAudioProcessor.from_pretrained(path)
    
    @staticmethod
    def load_audio(path):
        with suppress_logs(), suppress_output():
            audio, _ = librosa.load(path, sr=16000)
        return audio
    
    def load_flamingo_llm(self):
        with suppress_logs(), suppress_output():
            runner = self._get_sglang_runner()
        return runner
    
    def _get_sglang_runner(self, attention_backend="triton", tp_size=1):
        model_override = {"architectures": ["MusicFlamingoQwen2ForCausalLM"], "model_type": "qwen2"}
        server_args_kwargs = {
            "model_path": self.path,
            "tokenizer_path": self.path,
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
        return model_runner
    
    
    def prepare_model_input(self):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.describe_music_prompt},
                    {"type": "audio", "path":  "<sound>"},
                ],
            }
        ]
        text_prompt = self.music_processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        return text_prompt
    
        


















SGLANG_PYTHON_PATH = os.path.join(os.path.dirname(__file__), "sglang", "python")
if SGLANG_PYTHON_PATH not in sys.path:
    sys.path.insert(0, SGLANG_PYTHON_PATH)

_sglang_runner = None
_sglang_runner_path = None






def prepare_music_embedding(model, processor, music_path):
    

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



def inference_music_flamingo_lyrics(model, music_path, has_lyric_prompt, lyric_extract_prompt):
    # dont write this yet
    pass

def inference_qwen(model, text):
    # dont write this yet
    pass
