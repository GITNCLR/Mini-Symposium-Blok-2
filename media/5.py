# https://platform.openai.com/docs/guides/text-to-speech

from pathlib import Path
from openai import OpenAI

# Initialiseer de OpenAI client met je API-sleutel
client = OpenAI(api_key="Vul hier je API-sleutel in")

# Definieer het pad waar het gegenereerde spraakbestand wordt opgeslagen
speech_file_path = Path("output.wav")  # Bestand wordt opgeslagen in de huidige map

# Vraag de OpenAI API om de tekst om te zetten naar spraak
response = client.audio.speech.create(
    model="tts-1-hd",  # Gebruik het hoogkwaliteitsmodel van OpenAI voor tekst-naar-spraak
    voice="alloy",  # Beschikbare stemmen: alloy, ash, coral, echo, fable, onyx, nova, sage, shimmer
    input="De snelle vos sprong behendig over de luie hond, terwijl de regen zachtjes tegen het raam tikte en in de verte het onweer steeds luider begon te rommelen, alsof de natuur zelf een indrukwekkende symfonie wilde spelen."
)

# Schrijf de inhoud van de respons direct naar een bestand
speech_file_path.write_bytes(response.content)

# Geef een melding in de terminal dat het bestand is opgeslagen
print("Spraakbestand opgeslagen als output.wav")