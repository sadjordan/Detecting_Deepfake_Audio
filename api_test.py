import requests

url = "http://127.0.0.1:5000/predict"
wav_file_path = "recordings/This is not Morgan Freeman  -  A Deepfake Singularity.mp3"

with open(wav_file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print(response.status_code)
print(response.json())

# print(requests.get("http://127.0.0.1:5000/test").text)