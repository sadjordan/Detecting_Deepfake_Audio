import pandas as pd
import glob
import os

# Path to your cached parquet files
parquet_files = glob.glob("/Users/jordan/.cache/huggingface/hub/datasets--MLCommons--peoples_speech/snapshots/f10597c5d3d3a63f8b6827701297c3afdf178272/clean_sa/*.parquet", recursive=True)

print("Found parquet files:", parquet_files)

# Create output directory if it doesn't exist
os.makedirs("audio", exist_ok=True)

# Read and save audio files
count = 0
for pf in parquet_files:
    df = pd.read_parquet(pf)
    for _, row in df.iterrows():
        audio_bytes = row['audio']['bytes']
        audio_id = row['id']
        # Save as FLAC file
        out_path = os.path.join("audio", f"{audio_id}.flac")
        with open(out_path, "wb") as f:
            f.write(audio_bytes)
        count += 1
print(f"Saved {count} audio files to 'audio' directory.")