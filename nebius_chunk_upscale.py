import os
import shutil
import time
import subprocess
import requests
import boto3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()

# --- CONFIGURATION ---
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("ENDPOINT_ID")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "eu-north1")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "https://storage.eu-north1.nebius.cloud:443")

NEBIUS_INPUT_PREFIX = "inputs/"      # Folder in Nebius containing raw videos
NEBIUS_OUTPUT_PREFIX = "upscaled/"  # Folder in Nebius for finished videos

CHUNK_DURATION = 60                  # Duration of each segment in seconds
MAX_WORKERS = 5                      # Parallel jobs dispatched to RunPod

BASE_DIR = Path("./workspace_temp")
DOWNLOAD_DIR = BASE_DIR / "downloads"
CHUNK_DIR = BASE_DIR / "chunks"
UPSCALED_CHUNK_DIR = BASE_DIR / "upscaled_chunks"
OUTPUT_DIR = BASE_DIR / "outputs"

# Initialize Nebius S3 Client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
    endpoint_url=S3_ENDPOINT_URL
)

RUNPOD_HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json"
}


def download_from_nebius(s3_key: str, local_path: Path):
    """Downloads a file from Nebius S3 to local storage."""
    print(f"--> Downloading '{s3_key}' from Nebius...")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(S3_BUCKET, s3_key, str(local_path))


def upload_to_nebius(local_path: Path, s3_key: str) -> str:
    """Uploads a local file to Nebius S3 and returns a presigned URL."""
    s3_client.upload_file(str(local_path), S3_BUCKET, s3_key)
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": s3_key},
        ExpiresIn=3600
    )


def split_video(input_file: Path, segment_time: int) -> list[Path]:
    """Splits a video file into stream-copied segments using FFmpeg."""
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"--> Chunking '{input_file.name}' into {segment_time}s segments...")

    chunk_pattern = CHUNK_DIR / "chunk_%04d.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", str(input_file),
        "-c", "copy", "-map", "0",
        "-segment_time", str(segment_time),
        "-f", "segment", "-reset_timestamps", "1",
        str(chunk_pattern)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    chunks = sorted(list(CHUNK_DIR.glob("chunk_*.mp4")))
    print(f"--> Created {len(chunks)} chunk(s).")
    return chunks


def process_single_chunk(chunk_path: Path) -> Path:
    """Uploads chunk to Nebius, triggers RunPod endpoint with credentials, polls, and downloads result."""
    s3_chunk_key = f"temp_chunks/{chunk_path.name}"
    input_url = upload_to_nebius(chunk_path, s3_chunk_key)

    # Dispatch job to RunPod with credentials passed in payload
    run_endpoint = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
    payload = {
        "input": {
            "video_url": input_url,
            "scale": 4,
            "s3_bucket": S3_BUCKET,
            "s3_endpoint_url": S3_ENDPOINT_URL,
            "aws_access_key_id": AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
            "aws_region": AWS_REGION
        }
    }

    res = requests.post(run_endpoint, json=payload, headers=RUNPOD_HEADERS).json()
    job_id = res.get("id")
    if not job_id:
        raise RuntimeError(f"[{chunk_path.name}] Failed to start RunPod job: {res}")

    print(f"[{chunk_path.name}] Processing (Job ID: {job_id})...")

    # Poll job status until completion
    status_endpoint = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}"
    while True:
        status_res = requests.get(status_endpoint, headers=RUNPOD_HEADERS).json()
        status = status_res.get("status")

        if status == "COMPLETED":
            output_url = status_res["output"]["upscaled_video_url"]
            out_file_path = UPSCALED_CHUNK_DIR / f"upscaled_{chunk_path.name}"

            # Download upscaled chunk locally
            video_bytes = requests.get(output_url).content
            with open(out_file_path, "wb") as f:
                f.write(video_bytes)

            print(f"[{chunk_path.name}] Chunk upscaled successfully.")
            return out_file_path

        elif status in ["FAILED", "CANCELLED"]:
            raise RuntimeError(f"[{chunk_path.name}] Job failed: {status_res}")

        time.sleep(5)


def merge_videos(upscaled_chunks: list[Path], output_file: Path):
    """Concatenates upscaled segments back into a single video file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    concat_list = BASE_DIR / "concat_list.txt"

    with open(concat_list, "w") as f:
        for chunk in upscaled_chunks:
            f.write(f"file '{chunk.absolute()}'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy", str(output_file)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if concat_list.exists():
        concat_list.unlink()


def cleanup_temp_dirs():
    """Removes local temporary workspace folders."""
    if BASE_DIR.exists():
        shutil.rmtree(BASE_DIR)


def process_nebius_pipeline():
    # Fetch raw video list from Nebius S3
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=NEBIUS_INPUT_PREFIX)
    if "Contents" not in response:
        print(f"No video files found in bucket '{S3_BUCKET}' under prefix '{NEBIUS_INPUT_PREFIX}'.")
        return

    video_keys = [
        obj["Key"] for obj in response["Contents"]
        if obj["Key"].lower().endswith((".mp4", ".mov", ".mkv", ".avi"))
    ]

    print(f"Found {len(video_keys)} video(s) in Nebius storage to process.\n")

    for index, s3_video_key in enumerate(video_keys, start=1):
        filename = Path(s3_video_key).name
        print(f"=== [{index}/{len(video_keys)}] Processing Video: {filename} ===")

        cleanup_temp_dirs()
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        UPSCALED_CHUNK_DIR.mkdir(parents=True, exist_ok=True)

        # Step A: Download full video from Nebius
        local_raw_path = DOWNLOAD_DIR / filename
        download_from_nebius(s3_video_key, local_raw_path)

        # Step B: Split locally into 60s chunks
        chunks = split_video(local_raw_path, CHUNK_DURATION)

        # Step C: Dispatch chunks to RunPod in parallel
        print(f"--> Sending {len(chunks)} chunk(s) across {MAX_WORKERS} parallel workers...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            upscaled_chunks = list(executor.map(process_single_chunk, chunks))

        # Step D: Merge upscaled chunks into final video file
        local_final_path = OUTPUT_DIR / f"upscaled_{filename}"
        merge_videos(upscaled_chunks, local_final_path)

        # Step E: Upload final upscaled video back to Nebius
        nebius_output_key = f"{NEBIUS_OUTPUT_PREFIX}upscaled_{filename}"
        print(f"--> Uploading finished video to Nebius: '{nebius_output_key}'...")
        upload_to_nebius(local_final_path, nebius_output_key)

        print(f"=== Successfully Completed: {filename} ===\n")

    cleanup_temp_dirs()
    print("All bucket processing tasks complete!")


if __name__ == "__main__":
    process_nebius_pipeline()