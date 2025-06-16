
# 🎵📚 AI Multimodal Book & Music Recommender Chatbot

An intelligent **multi-modal recommender system** that suggests **books or music** based on:

- 🎙️ **Voice input** (speech, humming, or singing)
- 🎧 **Audio song recognition** (via Shazam)
- ✍️ **Typed queries** (mood, genre, themes, etc.)
- 🌤️ **Real-time weather** in your city

Built using **RAG (Retrieval-Augmented Generation)**, **speech/audio recognition**, **deep learning**, and **LLMs**, all wrapped in a beautiful Gradio web app.

---

## 🔧 Key Features

| Feature                       | Description |
|------------------------------|-------------|
| 🎙️ **Audio Transcription**     | Transcribes speech or humming using OpenAI Whisper |
| 🎵 **Music Identification**   | Identifies hummed/sung songs via Shazam |
| 💬 **Natural Language Input** | Understands mood, genres, artists, emotions |
| 🌦️ **Weather Context**        | Enhances music suggestions based on your city’s current weather |
| 🔍 **Semantic Retrieval**     | Uses FAISS + Sentence Transformers for similarity search |
| 🤖 **LLM Personalization**    | Refines recommendations with Mistral-7B via OpenRouter |
| 📚 **Book Suggestions**       | Fetches titles based on theme, mood, or user interest |
| 🎧 **Music Suggestions**      | Offers personalized tracks with tempo, valence, and genre filters |
| 🖼️ **Gradio Interface**       | Interactive no-code UI for all input modes |
| ⚙️ **Async Compatibility**    | `nest_asyncio` ensures Colab/Jupyter support |

---

## 🧠 Tech Stack

| Layer               | Tools/Models |
|--------------------|--------------|
| **UI**             | Gradio |
| **Audio Input**    | OpenAI Whisper, `pydub`, `torchaudio`, `librosa` |
| **Music ID**       | `shazamio` |
| **Text Embedding** | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| **Similarity Search** | FAISS |
| **Language Model** | Mistral-7B-Instruct via [OpenRouter](https://openrouter.ai) |
| **Recommender Type** | Hybrid: Content-Based + RAG + LLM |
| **Weather Data**   | [OpenWeatherMap API](https://openweathermap.org/api) |
| **Music Metadata** | Spotify API (`spotipy`) |

---

## 📂 Dataset Requirements

> Place these in your working directory:

- 📚 `books.csv`: Must include `title`, `authors`, `average_rating`, `publication_date`, `publisher`.
- 🎵 `spotify_songs.csv`: Must include `track_name`, `track_artist`, `track_album_release_date`, `playlist_genre`, `playlist_subgenre`, `valence`, `tempo`, `energy`.

---

## 🌐 Dataset Sources

- [Goodreads Books Dataset](https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks)
- [30,000 Spotify Songs Dataset](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs)

---

## 🔑 Required API Keys

Set your API keys in the script or environment variables:

```python
WEATHER_API_KEY = "your_openweathermap_api_key"
OPENROUTER_KEY = "your_openrouter_api_key"
SPOTIFY_CLIENT_ID = "your_spotify_client_id"
SPOTIFY_CLIENT_SECRET = "your_spotify_client_secret"
````

> 🎵 **Note:** Shazamio does not require an API key.

---

## ⚙️ Installation

Recommended: **Google Colab**, or use locally with Python ≥ 3.8.

```bash
pip install -q kaggle sentence-transformers faiss-cpu gradio openai pydub librosa torchaudio nest_asyncio spotipy git+https://github.com/openai/whisper.git shazamio
```

---

## 🚀 How It Works

1. **User Input**:

   * Voice input (speech/humming) is transcribed via Whisper.
   * Shazam detects the track (if hummed).
   * Typed queries describe mood, genre, etc.
   * City name adds live weather context.

2. **Retrieval**:

   * FAISS + MiniLM embedding retrieves top matching items from local datasets.

3. **LLM Processing**:

   * Retrieved results + context are passed to Mistral-7B for rich, coherent suggestions.

4. **Output**:

   * Returns a natural-language recommendation list for books and/or music.

---

## 🧪 Example Queries

| Input                            | Output                                       |
| -------------------------------- | -------------------------------------------- |
| `"lofi music for rainy days"`    | Tracks with calm tempo and chill vibes       |
| Hum a song into mic              | Identifies the song → similar tracks shown   |
| `"mystery novels like Sherlock"` | Books with detective & suspenseful themes    |
| `"sad songs" + city = Mumbai`    | Suggests emotional music matching local rain |

---

## 🖼️ Gradio UI Layout (Text + Voice + Weather)

```
+-------------------------------------------------------------+
| [Book] [Music]          Your City: [_____]                 |
|-------------------------------------------------------------|
| Type your mood:  [____________________________]            |
| Speak / Hum:      [🎙️ Upload or record audio]             |
|-------------------------------------------------------------|
|              [ Generate Recommendation ]                   |
|-------------------------------------------------------------|
| 🎯 AI Recommendations:                                     |
| "Try reading 'Kafka on the Shore' by Haruki Murakami..."   |
+-------------------------------------------------------------+
```

---

## ▶️ Launch the App

```python
demo.launch(debug=True)
```

---

## 📌 Future Enhancements

* ✅ Emotion detection from voice input
* ✅ Book-to-song and song-to-book mappings
* ⏳ Feedback learning for personalization
* ⏳ Mobile-friendly app & PWA support
* ⏳ Playlist generation + export to Spotify

---

## 📃 License

This project is licensed under the MIT License. Feel free to fork and contribute!
