import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import joblib
import uvicorn
from pathlib import Path
# FastAPI app
app = FastAPI()

# Use absolute paths for model files (Render runs from project root)
MODEL_DIR = Path(__file__).parent / "model"
TOKENIZER_PATH = MODEL_DIR
MODEL_PATH = MODEL_DIR
LABEL_MAP_PATH = MODEL_DIR / "label_mappings.pkl"
MOOD_SONG_MAP_PATH = MODEL_DIR / "mood_song_map.pkl"

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Load label mappings and mood-song map
label2id, id2label = joblib.load(LABEL_MAP_PATH)
mood_to_songs = joblib.load(MOOD_SONG_MAP_PATH)

# Mood remap for matching to available songs
mood_remap = {
    "joy": "happy",
    "love": "relaxed",
    "surprise": "neutral"
    # Add more mappings if needed
}

# Request schema
class MoodRequest(BaseModel):
    text: str

# Predict route
@app.post("/predict")
def predict_mood(req: MoodRequest):
    try:
        # Tokenize input
        inputs = tokenizer(
            req.text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()

        # Map prediction to label
        mood = id2label[pred_id]
        print("Predicted mood (raw):", mood)

        # Remap mood to match mood_song_map keys
        mood = mood_remap.get(mood, mood)
        print("Mapped mood for song lookup:", mood)

        # Fetch songs
        songs = mood_to_songs.get(mood, [])

        return {
            "predicted_mood": mood,
            "recommended_songs": songs[:10]  # Top 10 songs
        }
    
    except Exception as e:
        return {"error": str(e)}

# For local dev: uvicorn backend.main:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
