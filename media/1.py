#https://huggingface.co/parler-tts/parler-tts-mini-multilingual-v1.1

import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

model_name = "parler-tts/parler-tts-mini-multilingual-v1.1"

# Controleren of een GPU beschikbaar is, anders fallback naar CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Laad het ParlerTTS-model en verplaats het naar het geselecteerde apparaat (GPU of CPU)
model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)

# Laad de tokenizer voor tekst naar spraak
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-multilingual-v1.1")

# Laad de tokenizer voor de spraakbeschrijving
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

# Beschrijving van de spraakstijl (man, expressief, gematigde snelheid en toonhoogte)
description = (
    "Mark levert een licht expressieve en geanimeerde toespraak met een matige snelheid en toonhoogte."
    "De opname is van zeer hoge kwaliteit, waarbij de stem van de spreker helder en zeer dichtbij klinkt."
)

# Converteer de beschrijving naar tokens
input_ids = description_tokenizer(description, return_tensors="pt").input_ids.to(device)

# De tekst die omgezet wordt naar spraak
prompt = (
    "De snelle vos sprong behendig over de luie hond, terwijl de regen zachtjes tegen het raam tikte "
    "en in de verte het onweer steeds luider begon te rommelen, alsof de natuur zelf een indrukwekkende "
    "symfonie wilde spelen."
)

# Converteer de tekstprompt naar tokens
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Genereer de audio met het model
generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)

# Zet de gegenereerde audio om in een numpy-array en verwijder extra dimensies
audio_arr = generation.cpu().numpy().squeeze()

# Sla het gegenereerde audiofragment op als een WAV-bestand
output_filename = model_name.split("/")[1] + ".wav"
sf.write(output_filename, audio_arr, model.config.sampling_rate)