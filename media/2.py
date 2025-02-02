#https://huggingface.co/Kodamn47/speecht5_finetuned_facebook_voxpopuli_dutch

from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch

# Tekst die omgezet moet worden naar spraak
prompt = "De snelle vos sprong behendig over de luie hond, terwijl de regen zachtjes tegen het raam tikte en in de verte het onweer steeds luider begon te rommelen, alsof de natuur zelf een indrukwekkende symfonie wilde spelen."

# Naam van het model dat gebruikt wordt voor text-to-speech (TTS)
model_name = "Kodamn47/speecht5_finetuned_facebook_voxpopuli_dutch"

# Laad de TTS-pipeline met het opgegeven model
synthesiser = pipeline("text-to-speech", model=modelname, device="cpu")

# Laad de dataset met spreker-embeddings
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

# Selecteer een spreker-embedding uit de dataset en zet deze om naar een tensor
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Genereer spraak op basis van de ingevoerde tekst en de spreker-embedding
speech = synthesiser(prompt, forward_params={"speaker_embeddings": speaker_embedding})

# Sla het gegenereerde audiobestand op als een WAV-bestand
output_filename = model_name.split("/")[1] + ".wav"
sf.write(output_filename, speech["audio"], samplerate=speech["sampling_rate"])
