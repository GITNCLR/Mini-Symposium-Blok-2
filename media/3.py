# https://huggingface.co/procit008/training_tts_nl_v2

from transformers import AutoTokenizer, AutoModelForTextToWaveform
import torch
import soundfile as sf

# Laad de tokenizer en het model voor text-to-speech (TTS)
model_name = "procit008/training_tts_nl_v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTextToWaveform.from_pretrained(model_name)

# Tekst die omgezet moet worden naar spraak
prompt = "De snelle vos sprong behendig over de luie hond, terwijl de regen zachtjes tegen het raam tikte en in de verte het onweer steeds luider begon te rommelen, alsof de natuur zelf een indrukwekkende symfonie wilde spelen."

# Tokeniseer de tekst
inputs = tokenizer(prompt, return_tensors="pt")

# Genereer de spraakgolfvorm (waveform) met het model
with torch.no_grad():  # Schakel gradiëntberekening uit om geheugen te besparen
    speech = model(**inputs).waveform

# Definieer de samplefrequentie (deze moet overeenkomen met wat het model gebruikt)
sampling_rate = model.config.sampling_rate

# Sla de gegenereerde spraak op als een WAV-bestand
output_filename = model_name.split("/")[1] + ".wav"
sf.write(output_filename, speech.squeeze().numpy(), samplerate=sampling_rate)
