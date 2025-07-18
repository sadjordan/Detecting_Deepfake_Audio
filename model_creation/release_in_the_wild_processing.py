import pandas as pd
import librosa
import numpy as np
import os
import csv

# Path to metadata CSV and audio directory
metadata_csv = "/Users/jordan/Desktop/Live_Projects/Detecting_Deepfake_Audio/release_in_the_wild/meta.csv"
audio_dir = "release_in_the_wild"

# Output CSV for spoofed audio features
output_csv = os.path.join("audio_csv", "release_in_the_wild_converted.csv")

# Feature header
header = [
    "chroma_stft", "rms", "spectral_centroid", "spectral_bandwidth", "rolloff", "zero_crossing_rate"
] + [f"mfcc{i+1}" for i in range(20)] + ["LABEL"]

# Read metadata
df = pd.read_csv(metadata_csv)

with open(output_csv, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=header)
    writer.writeheader()

    for idx, row in df.iterrows():
        if str(row["label"]).strip().lower() == "spoof":
            wav_path = os.path.join(audio_dir, str(row["file"]))
            if not os.path.exists(wav_path):
                print(f"File not found: {wav_path}")
                continue
            try:
                y, sr = librosa.load(wav_path, sr=44100)
                window_size = int(sr * 0.2)  # 0.2 seconds
                num_windows = int(np.floor(len(y) / window_size))
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
                    features["LABEL"] = 0  # 0 for deepfakes
                    writer.writerow(features)
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")

print(f"Saved spoofed audio features as {output_csv}")