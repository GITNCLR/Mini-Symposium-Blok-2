import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.cache_data.clear()

# Display the banner image
st.image("media/banner.png", use_container_width=True)
# Titel en inleiding
st.title("Een Analyse van Non-Cloudbased Nederlandse Tekst-naar-Spraak Modellen")


st.markdown("""
De afgelopen jaren is **tekst-naar-spraak (TTS) technologie** sterk verbeterd, met name door cloudgebaseerde oplossingen zoals OpenAI’s **TTS-1-hd**. Deze systemen leveren hoogwaardige, natuurlijke spraak, maar brengen ook nadelen met zich mee: ze vereisen een internetverbinding, brengen privacyrisico’s met zich mee en kunnen kostbaar zijn bij intensief gebruik.

Voor veel toepassingen – zoals spraakassistenten, voorleesfuncties en toegankelijkheidstools – is een **offline TTS-model** een aantrekkelijk alternatief. Maar hoe goed presteren deze lokale modellen in vergelijking met cloudgebaseerde oplossingen?

Hoewel er steeds meer TTS-modellen beschikbaar zijn, is het merendeel niet specifiek voor het **Nederlands** ontwikkeld. Veel modellen zijn getraind op Engelstalige of meertalige datasets, waardoor hun prestaties in het Nederlands aanzienlijk slechter kunnen zijn. 

In deze blogpost worden specifiek **Nederlandse TTS-modellen die lokaal kunnen draaien** geanalyseerd en beoordeeld op **natuurlijkheid, verstaanbaarheid, audio-kwaliteit, stememotie** en **algemene tevredenheid**. OpenAI’s **TTS-1-hd** fungeert als benchmark om de kwaliteit van deze offline alternatieven objectief te kunnen vergelijken.
""")


############################################################################################################
# Waarom Non-Cloudbased TTS?
############################################################################################################
st.header("Waarom Non-Cloudbased TTS?")

st.markdown("""
Veel moderne TTS-modellen draaien in de cloud en profiteren van krachtige rekeninfrastructuur. Dit biedt voordelen zoals hoge geluidskwaliteit en snelle updates. Toch zijn er situaties waarin een **offline model** misschien de betere keuze is.  

### Voordelen van een offline TTS-model:
- **Privacy:** Gegevens blijven volledig lokaal, zonder dat gevoelige informatie naar externe servers wordt gestuurd.
- **Snelheid:** Geen vertraging door netwerkverbindingen, wat vooral belangrijk is voor real-time toepassingen.
- **Beschikbaarheid:** Werkt altijd, zelfs zonder internetverbinding of bij storingen in cloudservices.
- **Kostenbeheersing:** Geen terugkerende abonnementskosten of verbruikskosten per gegenereerd audiobestand.

Deze voordelen maken offline TTS-oplossingen bijzonder geschikt voor toepassingen zoals spraakassistenten en toegankelijkheidstools. Maar hoe goed presteren deze modellen vergeleken met cloudgebaseerde alternatieven?.
""")


############################################################################################################
# Enquête: Evaluatie van de TTS-modellen
############################################################################################################
st.header("Evaluatie van de TTS-modellen")

st.markdown("""
Deze blogpost bevat een evaluatie van verschillende **Nederlandse Text-to-Speech (TTS) modellen**. Zowel **recent geüploade modellen van Hugging Face** als het **OpenAI TTS-model** zijn getest. Dit laatste model fungeert als een **State-of-the-Art cloudreferentie** om de kwaliteit van offline modellen te vergelijken.

### Onderzoeksmethode

Voor een eerlijke vergelijking zijn de volgende stappen gevolgd:

- Vijf TTS-modellen getest: vier open-source modellen van Hugging Face en het OpenAI TTS-model. De selectie is gebaseerd op beschikbaarheid, recentheid en populariteit.
- Elk model heeft dezelfde testzin voorgelezen:

  > *De snelle vos sprong behendig over de luie hond, terwijl de regen zachtjes tegen het raam tikte en in de verte het onweer steeds luider begon te rommelen, alsof de natuur zelf een indrukwekkende symfonie wilde spelen.*

- Vijf evaluatiecriteria vastgesteld om de prestaties te beoordelen.


De beoordeling is uitgevoerd door een testgroep van **10 deelnemers** met behulp van de **Mean Opinion Score (MOS)**, een veelgebruikte methode voor de subjectieve evaluatie van TTS-modellen. Elk door een model gegenereerd geluidsfragment werd beoordeeld op vijf criteria, met scores variërend van 1 (slecht) tot 5 (uitstekend), zoals hieronder weergegeven.


| Categorie        | Betekenis | 5 (Beste) | 4 | 3 | 2 | 1 (Slechtste) |
|-----------------|------------|------------|--|--|--|--|
| **Natuurlijkheid** | Hoe menselijk klinkt de stem? | Zeer natuurlijk | Natuurlijk | Redelijk | Onnatuurlijk | Zeer onnatuurlijk |
| **Verstaanbaarheid** | Hoe goed is de spraak te begrijpen? | Perfect verstaanbaar | Goed | Redelijk | Moeilijk verstaanbaar | Onverstaanbaar |
| **Audio-kwaliteit** | Technische geluidskwaliteit (ruis, artefacten) | Uitstekend | Goed | Redelijk | Matig | Slecht |
| **Stememotie** | Hoe expressief en levendig is de stem? | Zeer expressief | Goed expressief | Redelijk | Minimale expressie | Geen expressie |
| **Algemene tevredenheid** | Hoe tevreden is de luisteraar? | Zeer tevreden | Tevreden | Gematigd tevreden | Ontevreden | Zeer ontevreden |

De evaluatie laat de sterke en zwakke punten van elk model zien. Deze inzichten helpen bij het kiezen van het meest geschikte model voor offline gebruik.
""")


############################################################################################################
# Resultaten en Analyse
############################################################################################################
st.header("Modellen en Prestaties")

# Definieer de evaluatiescores
data = {
    "Model": [
        "parler-tts-mini-multilingual-v1.1",
        "speecht5_finetuned_facebook_voxpopuli_dutch",
        "training_tts_nl_v2",
        "mms-tts-nld",
        "OpenAI-TTS-1-hd",
    ],
    "Natuurlijkheid": [5, 1, 2, 4, 4],
    "Verstaanbaarheid": [4, 3, 3, 4, 5],
    "Audio-kwaliteit": [3, 2, 4, 4, 5],
    "Stememotie": [4, 1, 2, 3, 5],
    "Algemene tevredenheid": [4, 2, 3, 4, 5],
}

df = pd.DataFrame(data)

# Modelbeschrijvingen
model_info = [
    (
        "parler-tts-mini-multilingual-v1.1",
        """
[`Parler-TTS Mini Multilingual v1.1`](https://huggingface.co/parler-tts/parler-tts-mini-multilingual-v1.1) is een geavanceerd tekst-naar-spraak (TTS) model dat in staat is om natuurlijke en hoogwaardige spraak te genereren in acht Europese talen: Engels, Frans, Spaans, Portugees, Pools, Duits, Italiaans en Nederlands. Het model is een uitbreiding van Parler-TTS Mini en is getraind op ongeveer 9.200 uur aan niet-Engelse data en 580 uur aan hoogwaardige Engelse data.

### Belangrijkste kenmerken

- **Multilingualiteit:** Het model kan spraak genereren in de bovengenoemde acht talen, wat het veelzijdig maakt voor verschillende toepassingen.
- **Aanpasbare spraakkenmerken:** Gebruikers kunnen via tekstprompts controle uitoefenen over verschillende spraakkenmerken, zoals geslacht van de spreker, spreektempo, toonhoogte en achtergrondgeluid.  
  Bijvoorbeeld:  
  > "Een vrouwelijke spreker levert een licht expressieve en geanimeerde toespraak met een matige snelheid en toonhoogte. De opname is van zeer hoge kwaliteit, met een heldere en nabije stem."
- **Sprekerconsistentie:** Het model is getraind met 16 verschillende sprekers, elk met unieke kenmerken. Gebruikers kunnen een specifieke spreker selecteren door de naam in de beschrijving te vermelden, zoals:  
  > "Mark levert een licht expressieve en geanimeerde toespraak met een matige snelheid en toonhoogte. De opname is van zeer hoge kwaliteit, waarbij de stem van de spreker helder en zeer dichtbij klinkt."
"""
    ),
    (
        "speecht5_finetuned_facebook_voxpopuli_dutch",
        """
Het model [`speecht5_finetuned_facebook_voxpopuli_dutch`](https://huggingface.co/Kodamn47/speecht5_finetuned_facebook_voxpopuli_dutch) is een versie van Microsoft’s SpeechT5-tekst-naar-spraak (TTS) model, specifiek gefinetuned voor de Nederlandse taal. Dit model is getraind op de Nederlandse subset van het VoxPopuli-dataset, een uitgebreide meertalige spraakcorpus samengesteld uit opnames van het Europees Parlement.

### Belangrijkste kenmerken

- **Modelbasis:** Het model is gebaseerd op [`microsoft/speecht5_tts`](https://huggingface.co/microsoft/speecht5_tts), een TTS-model ontwikkeld door Microsoft.
- **Fijngetuned voor Nederlands:** Door training op de Nederlandse data van VoxPopuli is het model geoptimaliseerd voor het genereren van natuurlijke en vloeiende Nederlandse spraak.
"""
    ),
    (
        "training_tts_nl_v2",
        """
Het model [`training_tts_nl_v2`](https://huggingface.co/procit008/training_tts_nl_v2) is een tekst-naar-spraak (TTS) model beschikbaar op Hugging Face. Het is geüpload door de gebruiker [Rajan mahato](https://huggingface.co/procit008). Er is geen informatie over het model te vinden, wel is het duidelijk dat de gebruiker nog druk bezig is met trainen aangezien er regelmatig nieuwe versies verschijnen.

### Belangrijkste kenmerken

- **Gebaseerd op de VITS-architectuur:** Het model maakt gebruik van variational inference en adversarial training voor hoogwaardige spraaksynthese.
- **Categorieën:** Geclassificeerd onder **"Text-to-Audio"**, **"Transformers"**, **"Safetensors"**, en **"vits"**.
"""
    ),
    (
        "mms-tts-nld",
        """
[`MMS-TTS-NLD`](https://huggingface.co/facebook/mms-tts-nld) is een tekst-naar-spraak model ontwikkeld door Meta als onderdeel van het **Massively Multilingual Speech (MMS) project**. Dit model is specifiek getraind voor de Nederlandse taal.

### Belangrijkste kenmerken

- **Onderdeel van het Massively Multilingual Speech (MMS) project:**  
  Het MMS-project is ontwikkeld door Meta en richt zich op spraaktechnologieën voor meer dan 1.100 talen.
"""
    ),
    (
        "OpenAI-TTS-1-hd",
        """
[`OpenAI TTS-1-hd`](https://platform.openai.com/docs/guides/text-to-speech) is een tekst-naar-spraak model ontwikkeld door OpenAI. Dit model is ontworpen om **natuurlijke en expressieve spraak** te genereren en wordt aangeboden via de **OpenAI API**. Het model is geoptimaliseerd voor **real-time toepassingen** en biedt een breed scala aan stemmen en instellingen. Opvallend is dat het model Nederlands spreekt met een Amerikaans accent.

### Belangrijkste kenmerken

- **Meerdere ingebouwde stemmen:**  
  OpenAI TTS beschikt over **zes ingebouwde stemmen**, waardoor gebruikers kunnen kiezen uit verschillende stemprofielen.
- **Real-time verwerking:**  
  Het model is ontworpen voor **directe spraakuitvoer**, waardoor het ideaal is voor **interactieve toepassingen** zoals virtuele assistenten en spraakinterfaces.
- **Eenvoudige integratie:**  
  OpenAI TTS kan eenvoudig worden geïntegreerd via de **OpenAI API**, met ondersteuning voor verschillende programmeertalen en platforms.
"""
    ),
]

# Lijst met evaluatiecategorieën
categories = df.columns[1:].tolist()
colors = ["red", "green", "blue", "orange", "purple"]

# Loop door de modellen en toon alles in één sectie per model
for i, (model_name, description) in enumerate(model_info):
    st.subheader(model_name)
    st.markdown(description)
    # Radar plot voor het model
    model_scores = df[df["Model"] == model_name].iloc[:, 1:].values.flatten().tolist()
    model_scores.append(model_scores[0])  # Sluit de cirkel

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=model_scores,
            theta=categories + [categories[0]],
            fill="toself",
            name=model_name,
            line=dict(color=colors[i]),
        )
    )

    # Layout aanpassen: vaste schaal 0 - 5
    fig.update_layout(
        title="",
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=False,
    )

    # Weergeven in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    st.audio(f"media/{i + 1}.wav", format="audio/wav", autoplay=False)
    with st.container():
        # Expander voor het tonen van een voorbeeldcodebestand (optioneel)
        try:
            with open(f"media/{i + 1}.py", "r") as f:
                code_content = f.read()

            with st.expander(f"Laat voorbeeldcode zien voor {model_name}"):
                st.code(code_content, language="python")

        except FileNotFoundError:
            pass  # Als er geen codebestand is, laat het dan gewoon weg

# Toon de evaluatiescores in een tabel onderaan
st.subheader("Overzicht van Model Scores")
st.dataframe(df)



############################################################################################################
# Conclusie en Aanbevelingen
############################################################################################################
st.header("Conclusie en Aanbevelingen")

st.markdown("""

**Algemene Rangschikking op Gemiddelde Score:**
1. **OpenAI-TTS-1-hd** (**4.8**)
2. **parler-tts-mini-multilingual-v1.1** (**4**)
3. **mms-tts-nld** (**3.8**)
4. **training_tts_nl_v2** (**2.8**)
5. **speecht5_finetuned_facebook_voxpopuli_dutch** (**1.8**)

**Beste Model:**
- [`OpenAI TTS-1-hd`](https://platform.openai.com/docs/guides/text-to-speech) behaalt de hoogste gemiddelde score van 4.8, met topwaarderingen op verstaanbaarheid, audiokwaliteit, stememotie en algemene tevredenheid. Dit resultaat is niet verrassend, aangezien OpenAI een van de marktleiders is. Toch is er ruimte voor verbetering: het model heeft een duidelijk Amerikaans accent, terwijl specifiek op het Nederlands getrainde modellen dit niet hebben. Daarnaast is het uitsluitend als betaald cloudmodel beschikbaar en kan het niet offline worden gebruikt.

**Beste Offline Model:**
- [`Parler-TTS Mini Multilingual v1.1`](https://huggingface.co/parler-tts/parler-tts-mini-multilingual-v1.1) heeft een hoge gemiddelde score van **4**. Een groot voordeel van dit model is dat het meertalig is en dat er veel controle is over de spraakkenmerken door middel van een extra prompt. Ook heeft het model geen last van een Amerikaans accent in tegenstelling tot het OpenAI-model.
Alleen de geluidskwaliteit scoort iets lager omdat het lijkt alsof de spreker ver van de microfoon staat. Mogelijk is hier met prompt engineering nog iets te verbeteren!


**Slechtste Presterende Model:**
- [`speecht5_finetuned_facebook_voxpopuli_dutch`](https://huggingface.co/Kodamn47/speecht5_finetuned_facebook_voxpopuli_dutch) scoort het laagst met een gemiddelde van **1.8**. Dit model presteert ondermaats in alle categorieën. Het lijkt erop dat het model nog verder gefinetuned moet worden om betere resultaten te behalen. Op Hugging Face staan meerdere varianten van dit model die nog wekelijks geupdate worden dus wellicht presteert een andere variant van dit model beter!


Wie op zoek is naar de beste spraakkwaliteit en geen bezwaar heeft tegen een cloudoplossing, kiest voor OpenAI-TTS-1-hd. Wil je echter een lokaal draaiend model met veel controle over de spraak, dan is parler-tts-mini-multilingual-v1.1 een uitstekende keuze.

De ontwikkeling van open-source en offline TTS-modellen gaat snel, het is interessant om te volgen hoe deze technologie zich verder zal verbeteren. Vooral voor het Nederlands is er nog veel ruimte voor groei en optimalisatie.
Voor nu bieden de beschikbare modellen al een goede basis voor verschillende toepassingen, van spraakassistenten tot toegankelijkheidstools.
""")

st.markdown(
    """
    **Door:** [Noah Christian Le Roy](https://nl.linkedin.com/in/noah-le-roy)
    """)
############################################################################################################
# Interactieve sectie
############################################################################################################
st.header("Links")
st.markdown(
    """
    - [Hugging Face TTS-modellen](https://huggingface.co/models?other=text-to-speech)
    - [TTS Arena - Evalueer Engelse taalmodellen in een Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena)
    
    """
)
