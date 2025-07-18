from flask import Flask, request, jsonify
import librosa
import numpy as np
import pandas as pd
import joblib
import os
import tempfile
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("deepfake_voice_detector.pkl")

#decompose the .wav into what we will feed the model
def extract_features(wav_path, fs=44100, window_size_sec=0.2):
    y, sr = librosa.load(wav_path, sr=fs)
    window_size = int(fs * window_size_sec)
    num_windows = int(np.floor(len(y) / window_size))
    rows = []
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        y_win = y[start:end]
        features = {
            "chroma_stft": np.mean(librosa.feature.chroma_stft(y=y_win, sr=sr)),
            "rms": np.mean(librosa.feature.rms(y=y_win)),
            "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y_win, sr=sr)),
            "spectral_bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y_win, sr=sr)),
            "rolloff": np.mean(librosa.feature.spectral_rolloff(y=y_win, sr=sr)),
            "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y_win)),
        }
        mfccs = librosa.feature.mfcc(y=y_win, sr=sr, n_mfcc=20)
        for j in range(20):
            features[f"mfcc{j+1}"] = np.mean(mfccs[j])
        rows.append(features)
    df = pd.DataFrame(rows)
    return df

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file.save(tmp.name)
        features_df = extract_features(tmp.name)
        os.unlink(tmp.name)
    #actual pred work
    preds = model.predict(features_df)
    return jsonify({'predictions': preds.tolist()})

@app.route('/test', methods=['GET'])
def test():
    return "API is working!"

if __name__ == '__main__':
    app.run(debug=True)