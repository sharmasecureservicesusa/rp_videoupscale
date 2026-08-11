import os
import subprocess
import requests
import asyncio
import boto3
import runpod
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Limit internal GPU execution concurrency if running on a single worker
executor = ThreadPoolExecutor(max_workers=5)

# Optional S3 setup for worker output upload
S3_BUCKET = os.getenv("S3_BUCKET", "your-s3-bucket-name")
s3_client = boto3.client("s3")

def upload_to_s3(local_path: str, filename: str) -> str:
    s3_key = f"outputs/{filename}"
    s3_client.upload_file(local_path, S3_BUCKET, s3_key)
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": s3_key},
        ExpiresIn=3600
    )

def process_video_sync(job_input: dict) -> dict:
    video_url = job_input.get("video_url")
    scale = job_input.get("scale", 4)
    job_id = job_input.get("job_id", "temp_job")

    input_path = f"/tmp/{job_id}_input.mp4"
    output_path = f"/tmp/{job_id}_output.mp4"

    try:
        # 1. Download chunk from S3
        response = requests.get(video_url, stream=True)
        with open(input_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # 2. Run Real-ESRGAN upscaling
        cmd = [
            "python", "-m", "realesrgan.utils",
            "-i", input_path,
            "-o", output_path,
            "-s", str(scale),
            "-n", "RealESRGAN_x4plus",
            "--model_path", "/app/weights/RealESRGAN_x4plus.pth"
        ]
        subprocess.run(cmd, check=True)

        # 3. Upload upscaled chunk back to S3
        output_url = upload_to_s3(output_path, f"{job_id}_upscaled.mp4")

        return {"upscaled_video_url": output_url}

    finally:
        # Cleanup temporary files
        if os.path.exists(input_path): os.remove(input_path)
        if os.path.exists(output_path): os.remove(output_path)

async def async_handler(job):
    job_input = job["input"]
    job_input["job_id"] = job.get("id", "job")
    
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, process_video_sync, job_input)
    return result

runpod.serverless.start({
    "handler": async_handler,
    "concurrency_modifier": lambda current: 5
})