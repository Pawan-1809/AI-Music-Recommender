import pandas as pd
from sklearn.cluster import KMeans
import joblib

# Load Spotify Dataset
df = pd.read_csv("ai-mood-music-recommender\data\song.csv")

# Select audio features
features = ['danceability', 'energy', 'valence', 'acousticness']
df_clean = df.dropna(subset=features)

# Normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df_clean[features])

# Cluster into 5 groups
kmeans = KMeans(n_clusters=5, random_state=42)
df_clean['cluster'] = kmeans.fit_predict(X)

# Map clusters to moods (heuristic based on valence + energy)
def map_cluster_to_mood(row):
    if row['valence'] > 0.6 and row['energy'] > 0.6:
        return "happy"
    elif row['valence'] < 0.4 and row['energy'] < 0.4:
        return "sad"
    elif row['valence'] > 0.6 and row['energy'] < 0.4:
        return "relaxed"
    elif row['valence'] < 0.4 and row['energy'] > 0.6:
        return "angry"
    else:
        return "neutral"

df_clean['mood'] = df_clean.apply(map_cluster_to_mood, axis=1)

# Save mood → songs mapping
mood_to_songs = df_clean.groupby('mood')['song_title'].apply(list).to_dict()
joblib.dump(mood_to_songs, "mood_song_map.pkl")

print("🎶 Mood-to-song map saved!")