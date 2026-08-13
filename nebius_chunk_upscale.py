import os
import shutil
import time
import subprocess
import requests
import boto3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("ENDPOINT_ID")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "eu-north1")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "https://storage.eu-north1.nebius.cloud:443")

NEBIUS_INPUT_PREFIX = "inputs/"
NEBIUS_OUTPUT_PREFIX = "upscaled/"

CHUNK_DURATION = 60
MAX_WORKERS = 5

BASE_DIR = Path("./workspace_temp")
DOWNLOAD_DIR = BASE_DIR / "downloads"
CHUNK_DIR = BASE_DIR / "chunks"
UPSCALED_CHUNK_DIR = BASE_DIR / "upscaled_chunks"
OUTPUT_DIR = BASE_DIR / "outputs"

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
    print(f"--> Downloading '{s3_key}' from Nebius...", flush=True)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(S3_BUCKET, s3_key, str(local_path))


def upload_to_nebius(local_path: Path, s3_key: str) -> str:
    s3_client.upload_file(str(local_path), S3_BUCKET, s3_key)
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": s3_key},
        ExpiresIn=3600
    )


def delete_from_nebius(s3_key: str):
    try:
        s3_client.delete_object(Bucket=S3_BUCKET, Key=s3_key)
    except Exception as e:
        print(f"--> Warning: Failed to delete '{s3_key}': {e}", flush=True)


def split_video(input_file: Path, segment_time: int) -> list[Path]:
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"--> Chunking '{input_file.name}' into {segment_time}s segments...", flush=True)

    chunk_pattern = CHUNK_DIR / "chunk_%04d.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", str(input_file),
        "-c", "copy", "-map", "0",
        "-segment_time", str(segment_time),
        "-f", "segment", "-reset_timestamps", "1",
        "-movflags", "+faststart",  # CRITICAL: Ensures valid moov atom headers for RunPod ffprobe
        str(chunk_pattern)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    chunks = sorted(list(CHUNK_DIR.glob("chunk_*.mp4")))
    print(f"--> Created {len(chunks)} chunk(s).", flush=True)
    return chunks


def process_single_chunk(chunk_path: Path) -> Path:
    s3_chunk_key = f"temp_chunks/{chunk_path.name}"
    input_url = upload_to_nebius(chunk_path, s3_chunk_key)

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

    print(f"[{chunk_path.name}] Dispatched to RunPod (Job ID: {job_id})", flush=True)

    status_endpoint = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}"
    start_time = time.time()
    max_wait_seconds = 720  # 12 minutes timeout per chunk

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            raise TimeoutError(f"[{chunk_path.name}] RunPod Job {job_id} timed out after {max_wait_seconds} seconds.")

        try:
            status_res = requests.get(status_endpoint, headers=RUNPOD_HEADERS, timeout=10).json()
            status = status_res.get("status")

            if status == "COMPLETED":
                output_url = status_res["output"]["upscaled_video_url"]
                out_file_path = UPSCALED_CHUNK_DIR / f"upscaled_{chunk_path.name}"

                video_bytes = requests.get(output_url, timeout=60).content
                with open(out_file_path, "wb") as f:
                    f.write(video_bytes)

                delete_from_nebius(s3_chunk_key)
                delete_from_nebius(f"outputs/{job_id}_upscaled.mp4")

                print(f"[{chunk_path.name}] Chunk upscaled successfully.", flush=True)
                return out_file_path

            elif status in ["FAILED", "CANCELLED"]:
                raise RuntimeError(f"[{chunk_path.name}] RunPod Job failed: {status_res}")

            elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                if int(elapsed) % 30 == 0:
                    print(f"[{chunk_path.name}] Status: {status} ({int(elapsed)}s elapsed)", flush=True)

        except requests.RequestException as e:
            print(f"[{chunk_path.name}] Network warning: {e}", flush=True)

        time.sleep(5)


def merge_videos(upscaled_chunks: list[Path], output_file: Path):
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
    if BASE_DIR.exists():
        shutil.rmtree(BASE_DIR)


def process_nebius_pipeline():
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=NEBIUS_INPUT_PREFIX)
    if "Contents" not in response:
        print(f"No video files found in '{S3_BUCKET}/{NEBIUS_INPUT_PREFIX}'.", flush=True)
        return

    video_keys = [
        obj["Key"] for obj in response["Contents"]
        if obj["Key"].rstrip("/").lower().endswith((".mp4", ".mov", ".mkv", ".avi"))
        and obj.get("Size", 0) > 0
    ]

    if not video_keys:
        print(f"No valid videos found under '{NEBIUS_INPUT_PREFIX}'.", flush=True)
        return

    print(f"Found {len(video_keys)} video(s) to process.\n", flush=True)

    for index, s3_video_key in enumerate(video_keys, start=1):
        clean_key = s3_video_key.rstrip("/")
        filename = Path(clean_key).name
        print(f"=== [{index}/{len(video_keys)}] Processing Video: {filename} ===", flush=True)

        cleanup_temp_dirs()
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        UPSCALED_CHUNK_DIR.mkdir(parents=True, exist_ok=True)

        local_raw_path = DOWNLOAD_DIR / filename
        download_from_nebius(s3_video_key, local_raw_path)

        chunks = split_video(local_raw_path, CHUNK_DURATION)

        print(f"--> Dispatching {len(chunks)} chunk(s) across {MAX_WORKERS} workers...", flush=True)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            upscaled_chunks = list(executor.map(process_single_chunk, chunks))

        local_final_path = OUTPUT_DIR / filename
        merge_videos(upscaled_chunks, local_final_path)

        nebius_output_key = f"{NEBIUS_OUTPUT_PREFIX}{filename}"
        print(f"--> Uploading finished video: '{nebius_output_key}'...", flush=True)
        upload_to_nebius(local_final_path, nebius_output_key)

        delete_from_nebius(s3_video_key)
        print(f"=== Successfully Completed & Removed Original: {filename} ===\n", flush=True)

    cleanup_temp_dirs()
    print("All tasks complete!", flush=True)


if __name__ == "__main__":
    process_nebius_pipeline()