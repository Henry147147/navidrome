import json
import os
import sys
from typing import List, Optional, Union

import librosa
import torch
import logging
from transformers.audio_utils import AudioInput, make_list_of_audio
from transformers.processing_utils import Unpack
from transformers import MusicFlamingoForConditionalGeneration, MusicFlamingoProcessor, MusicFlamingoForConditionalGeneration
from transformers.models.musicflamingo.processing_musicflamingo import MusicFlamingoProcessorKwargs
from contextlib import contextmanager, redirect_stdout, redirect_stderr

DESCRIBE_PROMPT = "Describe this track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, and overall mood it creates."
MAX_NEW_TOKENS = 2048
class CustomMusicFlamingo(MusicFlamingoForConditionalGeneration):
    pass

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
    def __init__(self, path, device="cuda"):
        self.path = path
        self.device = torch.device(device)
        self.music_flamingo = MusicFlamingo.load_music_flamingo(self.path, device_map=self.device)
        self.music_processor = MusicFlamingo.load_music_processor(self.path)
        #self.llm = self.load_flamingo_llm()
        self.describe_music_prompt = DESCRIBE_PROMPT
        self.max_new_tokens = 2048
        
    def embed_music_from_path(self, path):
        loaded = self.load_audio(path)
        prepared = self.prepare_music(loaded)
        embedded = self.embed_music(prepared)
        return embedded.to("cpu").detach()
    
    def inference_llm(self, audio_embeds):
        text = self.prepare_model_input()
        inputs = self.music_processor(
                    text=text,
                    audio=None,
                    return_tensors="pt") # type: ignore
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        inputs_embeds = self.music_flamingo.get_input_embeddings()(input_ids)
        audio_token_mask = (input_ids == self.music_flamingo.config.audio_token_id).unsqueeze(-1)
        
        inputs_embeds = inputs_embeds.masked_scatter(
            audio_token_mask.to(inputs_embeds.device), audio_embeds.to(inputs_embeds.device)
        )
        outputs = self.music_flamingo.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS
        )
        return outputs

        
        

    def prepare_music(self, audio):
        processed = self.music_processor(None, audio, return_tensors="pt").to(self.device) # type: ignore
        del processed["input_ids"]
        del processed["attention_mask"]
        return processed
    
    def embed_music(self, inputs):
        with torch.inference_mode():
            embeddings = self.music_flamingo.get_audio_features(**inputs)
        return embeddings
        
    @staticmethod
    def load_music_flamingo(path, **kwargs):
        with suppress_logs(), suppress_output():
            return CustomMusicFlamingo.from_pretrained(path, **kwargs).eval()
    
    @staticmethod
    def load_music_processor(path):
        return MusicFlamingoAudioProcessor.from_pretrained(path)
    
    @staticmethod
    def load_audio(path):
        with suppress_logs(), suppress_output():
            audio, _ = librosa.load(path, sr=16000)
        return audio

    
    
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
            tokenize=False, # type: ignore
            add_generation_prompt=True, # type: ignore
        )
        
        return text_prompt
    
def test():
    songs = ["/home/henry/projects/navidrome/music/Lorde-Pure_Heroine-24BIT-WEB-FLAC-2013-TVRf/04-lorde-ribs.flac",
        "/home/henry/projects/navidrome/music/Lorde-Pure_Heroine-24BIT-WEB-FLAC-2013-TVRf/01-lorde-tennis_court.flac"]
    prompt = "Describe this track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, and overall mood it creates."
    mf = MusicFlamingo("./music_flamingo_fp8")
    embedded = mf.embed_music_from_path(songs[0])
    inference = mf.inference_llm(embedded)
    print(inference)




if __name__ == "__main__":
    test()
