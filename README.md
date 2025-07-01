# 🎵 AI Mood Music Recommender 🎧  
_Your personal mood-based music buddy powered by BERT, FastAPI, and Streamlit_

## ✨ What is this?

This project uses **AI to detect the mood** from your input text and recommends songs based on the detected emotion. Whether you're feeling happy, sad, in love, or surprised — it finds songs to vibe with your mood 🎶

Built using:

- 🤖 BERT for mood classification
- 🚀 FastAPI for serving the backend API
- 🌐 Streamlit for a clean, interactive frontend
- 🎼 A mood-tagged song dataset
- ☁️ Render for cloud deployment

## 📸 Demo

**You type:**  "I'm feeling so relaxed and calm today."


**You get:**  
`Mood: Relaxed`  
`🎶 Recommended Songs:`  
- Weightless – Marconi Union  
- Strawberry Swing – Coldplay  
- ...and more!

## 🛠️ Tech Stack

| Tool               | Use                                      |
|--------------------|-------------------------------------------|
| `transformers`     | BERT-based mood classification            |
| `FastAPI`          | Backend REST API                          |
| `Streamlit`        | Frontend interface                        |
| `joblib`           | Save/load mood mappings & song data       |
| `Render`           | Deploying the FastAPI backend             |
| `Hugging Face`     | Model + tokenizer hub                     |

## 🚀 Getting Started (Run Locally)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/ai-mood-music-recommender.git
cd ai-mood-music-recommender

