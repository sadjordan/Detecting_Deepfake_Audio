import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import pandas as pd
import os
import noisereduce as nr

fs = 44100
output_wav = "recordings/output.wav"
output_csv = "recordings/output.csv"

duration = None  # We'll record until user presses Enter

print("Press Enter to stop recording...")

# Start recording
recording = []
def record_audio():
    with sd.InputStream(samplerate=fs, channels=1, dtype='int16') as stream:
        while not stop_flag[0]:
            data, _ = stream.read(1024)
            recording.append(data)

stop_flag = [False]
import threading
record_thread = threading.Thread(target=record_audio)
record_thread.start()

input()  # Wait for Enter
stop_flag[0] = True
record_thread.join()

print("Recording finished.")

# Concatenate all recorded chunks
audio = np.concatenate(recording, axis=0)

# Save as WAV
os.makedirs("recordings", exist_ok=True)
write(output_wav, fs, audio)
print("Saved as output.wav")

# Load audio for feature extraction (librosa expects float32)
y, sr = librosa.load(output_wav, sr=fs)

# Reduce noise
y = nr.reduce_noise(y=y, sr=sr)

window_size = int(fs * 0.1)  # 0.1 secs per row
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

