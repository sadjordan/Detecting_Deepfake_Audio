# Deepfake Audio Detection API

## How to Call the API

### 1. Start the Flask API Server

Make sure you have all dependencies installed (`pip install -r requirements.txt`) and run:

```sh
python deepfake_detector_API.py
```

The API will be available at `http://127.0.0.1:5000`.

---

### 2. Send an Audio File for Prediction

You can use Python's `requests` library or `curl` to send a `.wav` file to the API.

#### Example using Python (`api_test.py`):

```python
import requests

url = "http://127.0.0.1:5000/predict"
wav_file_path = "recordings/your_audio.wav"

with open(wav_file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print(response.status_code)
print(response.json())
```

#### Example using curl:

```sh
curl -X POST -F "file=@recordings/your_audio.wav" http://127.0.0.1:5000/predict
```

The API will return a JSON object with predictions for each window of the audio file.

---

### 3. Test the API Server

You can check if the server is running with:

```python
import requests
print(requests.get("http://127.0.0.1:5000/test").text)
```

---

## Project Overview

### Data Collection & Preparation

- Audio datasets were collected and labeled as "REAL" or "FAKE".
- Feature extraction was performed using `librosa` to compute chroma, RMS, spectral features, rolloff, zero crossing rate, and MFCCs for short windows of each audio file.
- Data was saved in CSV format for model training.

### Model Training

- Multiple models were trained and evaluated, including KNN, Naive Bayes, Random Forest, and XGBoost.
- The final model was trained using XGBoost for binary classification.
- The trained model was saved as `deepfake_voice_detector.pkl`.

### API Development

- The Flask API (`deepfake_detector_API.py`) accepts `.wav` files via POST requests.
- The API extracts features from the uploaded audio, feeds them into the trained model, and returns predictions for each window.
- CORS is enabled for cross-origin requests.
- The `/test` endpoint is available for health checks.

### Usage

- Record or upload an audio file.
- Send the file to the API for deepfake detection.
- Receive a prediction for each window (e.g., `0` for fake, `1` for real, or `*` for quiet sections).

---

## Folder Structure

- `deepfake_detector_API.py` — Flask API server
- `api_test.py` — Example client for API testing
- `recordings/` — Folder for input/output audio and CSV files
- `model_creation/` — Scripts for data processing and model training
- `deepfake_voice_detector.pkl` — Trained model

---

## Notes

- Ensure your audio files are in `.wav` format for best compatibility.

---