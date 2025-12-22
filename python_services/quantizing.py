from transformers import (
    AudioFlamingo3ForConditionalGeneration,
    AutoProcessor as FlamingoProcessor,
)
import tqdm
from glob import glob
import torch
from optimum.quanto import quantize, qfloat8, Calibration, freeze
from optimum.quanto.tensor.qtype import qfloat

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
quantize(model, weights=qfloat8, activations=qfloat(torch.bfloat16), include=include)

flacs = list(glob("/mnt/z/music/**/*.flac"))
mp3s = list(glob("/mnt/z/music/**/*.mp3"))

all_songs = []
all_songs.extend(flacs)
all_songs.extend(mp3s)

processor = FlamingoProcessor.from_pretrained("nvidia/music-flamingo-hf")

import random
from typing import Sequence

_MUSIC_FLAMINGO_PROMPTS: Sequence[str] = [
    """Describe this track in full detail: identify the genre/subgenre, approximate tempo (BPM), and musical key (or tonal center).
Then describe instrumentation and timbre, the groove/rhythm feel, arrangement/structure (intro/verse/chorus/bridge), and how the energy evolves over time.
Include production and mix details (space, saturation, compression, stereo image), and the overall mood/scene/era it evokes.""",
    """Write a rich caption that blends technical details (genre, BPM, key/mode, chords or harmonic motion, mix/production) with how the song feels emotionally as it unfolds.
Call out the main sections and what changes in each (new layers, drops, modulation, rhythmic switches).""",
    """Break the track down like a critic: list tempo (BPM), key/tonal center, and the main chord progression(s) if you can infer them.
Then explain the textures (lead lines, pads, bass, drums), dynamics, and the emotional impact of the performance and production choices.""",
    """Provide a structured analysis with headings:
1) Genre & influences (and likely era/scene),
2) Rhythm & tempo (BPM) + time-feel (swing/straight/half-time),
3) Harmony (key/mode + notable chord colors),
4) Melody & hooks,
5) Arrangement (section map),
6) Sound design & mix,
7) Mood and narrative arc.""",
    """Analyze the vocals (if present): gender/timbre, style (spoken/sung/rap), phrasing, harmonies/doubles, and effects (autotune, distortion, reverb/delay).
Describe the language; if lyrics are clearly intelligible, quote only short recurring phrases and summarize the themes—otherwise state that they aren't clearly intelligible.""",
    """Focus on instrumentation and production: identify the core instruments/synths/drum sounds and how they're layered.
Describe the drum programming (kick/snare/hat patterns), bass design (808, electric, synth), and any signature ear-candy (risers, chops, fills).
Conclude with what the mix/master prioritizes (punch, warmth, brightness, loudness, width).""",
    """Give a 'DJ-friendly' breakdown: estimate BPM, key, and describe the beat pattern and drop/peak moments.
Explain where the track builds, where it releases, and what sections would be best for mixing (intros/outros/breakdowns).""",
    """Do a harmony-and-form deep dive: identify tonal center/key, any mode mixture, cadences, and repeating progressions.
Map the structure (timestamps if possible) and explain how harmonic/rhythmic changes support the emotional arc.""",
    """Describe the track in terms of mood and cinematic imagery, but ground it in musical evidence:
what specific instruments, harmonies, rhythms, and production choices create that mood?
Also note any cultural/genre markers that place it in a particular scene or regional style.""",
    """Compare-and-contextualize: describe the genre/BPM/key and then name a few *types* of artists/scenes it resembles (without guessing the exact song/artist unless it's obvious).
Explain the similarities in sound palette, vocal treatment, groove, and arrangement conventions.""",
    """Describe this track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, 
            and overall mood it creates. Also describe the language used (or if it is instrumental), any clearly intelligible and 
            important lyrics or recurring phrases and what they are about, the emotions the music and vocals evoke and the era or scene it most strongly resembles. Include any 
            other musically relevant details such as how the energy changes over time.""",
            "Describe this track in full detail"
]


def random_music_flamingo_prompt() -> str:
    """
    Returns a random Music Flamingo-style instruction prompt for describing an input song.
    If seed is provided, selection is deterministic.
    """
    return random.choice(_MUSIC_FLAMINGO_PROMPTS)


def prepare(processor, audio_path: str) -> dict:
    """
    Generate a rich caption for the provided audio file using the official
    Music Flamingo chat template, mirroring the model card example.
    """
    prompt = random_music_flamingo_prompt()
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

print("starting calibration")

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

save_file(model.state_dict(), "music_flamingo_fp8.safetensor")

import json

from optimum.quanto import quantization_map

with open('music_flamingo_fp8_quantization_map.json', 'w') as f:
  json.dump(quantization_map(model), f)
