from transformers import (
    AutoModelForCausalLM,
    MusicFlamingoForConditionalGeneration,
    AutoProcessor,
    FineGrainedFP8Config,
    TextStreamer
)

def load_music_flamingo(path):
    pass

def load_qwen_embedder(path):
    pass

def prepare_music(preprocessor, music_path):
    pass

def inference_music_flamingo(model, music_path, prompt, embedding=True): 
    pass

def inference_music_flamingo_lyrics(model, music_path, has_lyric_prompt, lyric_extract_prompt):
    pass

def inference_qwen(model, text):
    pass


