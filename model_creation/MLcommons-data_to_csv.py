import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
import glob
import os
import csv

fs = 44100
audio_dir = "audio"
output_dir = "audio_csv"
os.makedirs(output_dir, exist_ok=True)

audio_files = glob.glob(os.path.join(audio_dir, "*.flac"))

output_csv = os.path.join(output_dir, "all_audio_features.csv")

# Write header first
header = [
    "chroma_stft", "rms", "spectral_centroid", "spectral_bandwidth", "rolloff", "zero_crossing_rate"
] + [f"mfcc{i+1}" for i in range(20)] + ["LABEL"]

with open(output_csv, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=header)
    writer.writeheader()

    for audio_path in audio_files:
        try:
            y, sr = librosa.load(audio_path, sr=fs)
            window_size = int(fs * 0.2)  # 0.2 seconds
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
                features["LABEL"] = 1
                writer.writerow(features)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

print(f"Saved all features as {output_csv}")

# so you know it's done
data, samplerate = sf.read("audio/2016_Princeton_Board_of_Education_Candidates_Forum_SLASH_2016_Princeton_Board_of_Education_Candidates_Forum_DOT_mp3_00052.flac.flac")
sd.play(data, samplerate)
sd.wait()

