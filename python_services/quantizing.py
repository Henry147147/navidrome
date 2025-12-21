from transformers import (
    AudioFlamingo3ForConditionalGeneration,
    AutoProcessor as FlamingoProcessor,
)
import tqdm
from glob import glob
import torch
from optimum.quanto import quantize, qfloat8, Calibration, freeze

include = [
    # Attention projections
    "language_model.model.layers.*.self_attn.q_proj",
    "language_model.model.layers.*.self_attn.k_proj",
    "language_model.model.layers.*.self_attn.v_proj",
    "language_model.model.layers.*.self_attn.o_proj",
    # MLP projections
    "language_model.model.layers.*.mlp.gate_proj",
    "language_model.model.layers.*.mlp.up_proj",
    "language_model.model.layers.*.mlp.down_proj",
]

model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
    "nvidia/music-flamingo-hf", device_map="auto", torch_dtype=torch.bfloat16
)
print(model.dtype)
quantize(model, weights=qfloat8, activations=qfloat8, include=include)

flacs = list(glob("/mnt/z/music/**/*.flac"))
mp3s = list(glob("/mnt/z/music/**/*.mp3"))

all_songs = []
all_songs.extend(flacs)
all_songs.extend(mp3s)

processor = FlamingoProcessor.from_pretrained("nvidia/music-flamingo-hf")


def prepare(processor, audio_path: str) -> dict:
    """
    Generate a rich caption for the provided audio file using the official
    Music Flamingo chat template, mirroring the model card example.
    """
    prompt = """Describe this track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, 
            and overall mood it creates. Also describe the language used (or if it is instrumental), any clearly intelligible and 
            important lyrics or recurring phrases and what they are about, the emotions the music and vocals evoke and the era or scene it most strongly resembles. Include any 
            other musically relevant details such as how the energy changes over time."""

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": str(audio_path)},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )

    return inputs


with Calibration():
    for song in tqdm.tqdm(all_songs):
        prepared = prepare(processor, song)
        for k, v in list(prepared.items()):
            if isinstance(v, torch.Tensor):
                if torch.is_floating_point(v):
                    prepared[k] = v.to(torch.bfloat16).to(model.device)
                else:
                    prepared[k] = v.to(model.device)

        model.generate(**prepared, max_new_tokens=8192)

print("done calibrating")

freeze(model)

from safetensors.torch import save_file

save_file(model.state_dict(), "nvidia-music-flamingo-hf-fp8.safetensor")
