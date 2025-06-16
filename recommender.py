!pip install -q kaggle sentence-transformers faiss-cpu gradio openai pydub librosa torchaudio git+https://github.com/openai/whisper.git
from google.colab import files
files.upload()
!mv "kaggle (2).json" kaggle.json
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d jealousleopard/goodreadsbooks
!unzip -q goodreadsbooks.zip
!kaggle datasets download -d joebeachcapital/30000-spotify-songs
!unzip -q 30000-spotify-songs.zip

import nest_asyncio
import asyncio
import gradio as gr
import tempfile
import numpy as np
import soundfile as sf
import pandas as pd
import faiss
import requests
import whisper
from shazamio import Shazam
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# 🔁 Allow nested asyncio in notebooks/Gradio
nest_asyncio.apply()
shazam = Shazam()

# 🔑 API keys
WEATHER_API_KEY = "00548d03547fb776447f9cb636fbe753"
OPENROUTER_KEY = "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"
SPOTIFY_CLIENT_ID = "5f86073d20a84b1082632bd3304f02ab"
SPOTIFY_CLIENT_SECRET = "e977c9acfbb34c00b9af1f37f945f91b"

# 🎧 Spotify Client
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET
))

# 🧠 OpenRouter LLM client
client = OpenAI(
    api_key=OPENROUTER_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# 📚 Load datasets
books_df = pd.read_csv("books.csv", on_bad_lines='skip', encoding='utf-8')
music_df = pd.read_csv("spotify_songs.csv", encoding='utf-8')

books_df = books_df[['title', 'authors', 'average_rating', 'publication_date', 'publisher']].dropna()
music_df = music_df[['track_name', 'track_artist', 'track_album_release_date', 'playlist_genre', 'playlist_subgenre', 'valence', 'tempo', 'energy']].dropna()

books_df['combined'] = books_df.apply(
    lambda r: f"'{r['title']}' by {r['authors']} ({r['publication_date']}, {r['publisher']}) - Rating: {r['average_rating']}.", axis=1
)
music_df['combined'] = music_df.apply(
    lambda r: f"{r['track_name']} by {r['track_artist']} ({r['track_album_release_date']}) - Genre: {r['playlist_genre']}/{r['playlist_subgenre']}, Valence: {r['valence']}, Energy: {r['energy']}, Tempo: {r['tempo']}", axis=1
)

# 🔍 Sentence Transformer + FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
book_embeds = model.encode(books_df['combined'].tolist(), show_progress_bar=True)
music_embeds = model.encode(music_df['combined'].tolist(), show_progress_bar=True)

book_index = faiss.IndexFlatL2(book_embeds.shape[1])
book_index.add(np.array(book_embeds))
music_index = faiss.IndexFlatL2(music_embeds.shape[1])
music_index.add(np.array(music_embeds))

# 🎙️ Whisper model
asr_model = whisper.load_model("base")

# 🌦️ Weather API
def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    res = requests.get(url).json()
    if 'weather' in res:
        return res['weather'][0]['description']
    return ""

# 🎧 Shazam Song Identifier (Async)
async def identify_song_async(path):
    out = await shazam.recognize(path)
    try:
        title = out['track']['title']
        artist = out['track']['subtitle']
        return f"{title} by {artist}"
    except:
        return None

def identify_song_sync(path):
    try:
        return asyncio.get_event_loop().run_until_complete(identify_song_async(path))
    except RuntimeError:
        # In case of event loop already running (like Jupyter), use new loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(identify_song_async(path))

# 📢 Speech/Humming Transcription
def transcribe(path):
    return asr_model.transcribe(path)["text"]

# 📚 Book Similarity
def get_similar_books(query, k=5):
    vec = model.encode([query])
    _, I = book_index.search(np.array(vec), k)
    return books_df.iloc[I[0]]

# 🎵 Music Similarity
def get_similar_music(query, k=5):
    vec = model.encode([query])
    _, I = music_index.search(np.array(vec), k)
    return music_df.iloc[I[0]]

# 💡 LLM Recommendation
def ask_llm(context, query, domain):
    prompt = f"You are a helpful assistant recommending {domain}s.\nUser query: {query}\nCandidates:\n{context}\nGive personalized and unique recommendations."
    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# 🧠 Main Recommender Function
def recommend(text, audio, task, city):
    query = text or ""
    song_info = ""

    if audio is not None:
        try:
            # 📁 Case 1: File was uploaded via Gradio (audio is file path string)
            if isinstance(audio, str):
                temp_audio_path = audio

            # 🎙️ Case 2: Recorded via microphone (audio is (numpy_array, sample_rate))
            elif isinstance(audio, tuple) and len(audio) == 2:
                audio_data, sample_rate = audio
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    audio_data = np.atleast_2d(audio_data)
                    if audio_data.shape[0] > audio_data.shape[1]:
                        audio_data = audio_data.T
                    sf.write(f.name, audio_data.T, sample_rate)
                    temp_audio_path = f.name

            else:
                return "⚠️ Invalid audio input format"

            # 🧠 Transcribe speech or humming
            transcribed = transcribe(temp_audio_path)
            query += " " + transcribed

            # 🎵 Identify the song using Shazam
            song = identify_song_sync(temp_audio_path)
            if song:
                query += f" similar to {song}"
                song_info = f"🎵 Identified: {song}\n\n"

        except Exception as e:
            return f"⚠️ Error processing audio: {str(e)}"

    # 🌤️ Add weather context
    if city:
        weather = get_weather(city)
        if weather:
            query += f" weather: {weather}"

    # 📚 Book Recommendation
    if task == "book":
        results = get_similar_books(query)
        context = '\n'.join(
            f"{r.title} by {r.authors} ({r.publication_date}) - Rating: {r.average_rating}"
            for _, r in results.iterrows()
        )
        recs = ask_llm(context, query, "book")

    # 🎵 Music Recommendation
    else:
        results = get_similar_music(query)
        context = '\n'.join(
            f"{r['track_name']} by {r['track_artist']} ({r['track_album_release_date']}) - Genre: {r['playlist_genre']}"
            for _, r in results.iterrows()
        )
        recs = ask_llm(context, query, "music")

    return song_info + recs

# 🧪 Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎵📖 AI Recommender Chatbot — Books, Music, Mood, Weather, Humming (Shazam Only)")

    task = gr.Radio(["book", "music"], label="What would you like?", value="book")
    city = gr.Textbox(label="Your City (for climate-based music)", value="")
    txt = gr.Textbox(label="Type your mood, artist, genre, time period etc.")
    aud = gr.Audio(label="🎙️ Speak / hum / describe", type="filepath", interactive=True)
    out = gr.Textbox(label="🎯 AI Recommendations")

    btn = gr.Button("Suggest")
    btn.click(fn=recommend, inputs=[txt, aud, task, city], outputs=out)

    gr.Markdown("### 🟢 You can speak, type or hum to get music + book suggestions. Songs are identified using Shazam 🎧")

demo.launch(debug=True)
