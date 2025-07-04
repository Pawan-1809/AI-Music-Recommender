import streamlit as st
import requests
import os

st.set_page_config(page_title="AI Mood Music Recommender 🎵", page_icon="🎧")
st.markdown(
    "<h1 style='text-align: center; color: #1DB954;'>🎶 AI Mood Music Recommender</h1>",
    unsafe_allow_html=True
)
st.write("Tell me how you're feeling in words, and I'll give you songs that match your vibe!")

# User input
user_input = st.text_area("💬 What's on your mind?", height=150)

mood_emojis = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😡",
    "relaxed": "😌",
    "neutral": "🙂"
}

# Use Render environment variable or fallback to localhost for dev
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:10000")

if st.button("🔍 Predict Mood"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        try:
            # Send request to FastAPI
            response = requests.post(
                f"{BACKEND_URL}/predict",
                json={"text": user_input}
            )

            if response.status_code == 200:
                data = response.json()
                emoji = mood_emojis.get(data['predicted_mood'].lower(), "")
                st.success(f"🎭 Predicted Mood: *{data['predicted_mood'].capitalize()}* {emoji}")
                st.markdown("🎧 *Recommended Songs:*")
                for i, song in enumerate(data['recommended_songs'], 1):
                    # Generate a YouTube search URL
                    youtube_url = f"https://www.youtube.com/results?search_query={'+'.join(song.split())}+lyrics"
                    st.markdown(f"{i}. [{song}]({youtube_url}) 🔗")
            else:
                st.error("Something went wrong with the prediction.")

        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to the FastAPI backend. Is it running?")