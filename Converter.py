# import sounddevice as sd
# from scipy.io.wavfile import write
import librosa
import numpy as np
import pandas as pd
# import os
# import noisereduce as nr

duration = 5  # seconds
fs = 44100
input_wav = "/Users/jordan/.cache/kagglehub/datasets/birdy654/deep-voice-deepfake-voice-recognition/versions/2/KAGGLE/AUDIO/REAL/taylor-original.wav"
output_csv = "recordings/output_taylor_real.csv"

y, sr = librosa.load(input_wav, sr=fs)

window_size = fs  # 1 second
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
df.to_csv(output_csv, index=False)
print(f"Saved features as {output_csv}")