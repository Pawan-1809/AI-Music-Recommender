from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import joblib
import uvicorn

# FastAPI app
app = FastAPI()

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("backend/model/model")
model = BertForSequenceClassification.from_pretrained("backend/model/model")
model.eval()

# Load label mappings and mood-song map
id2label, label2id = joblib.load("backend/model/label_mappings.pkl")  # FIXED: Correct unpacking
mood_to_songs = joblib.load("backend/model/mood_song_map.pkl")

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

