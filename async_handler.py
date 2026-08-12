import os
import shutil
import subprocess
import requests
import asyncio
import boto3
import torch
import cv2
import runpod
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Process GPU tasks
executor = ThreadPoolExecutor(max_workers=1)

# Nebius S3 Config
S3_BUCKET = os.getenv("S3_BUCKET")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "https://storage.eu-north1.nebius.cloud:443")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "eu-north1")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
    endpoint_url=S3_ENDPOINT_URL
)

# Load Real-ESRGAN Model into GPU VRAM on worker startup
print("Loading Real-ESRGAN Model into VRAM...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path="/app/weights/RealESRGAN_x4plus.pth",
    model=model,
    tile=400,        # Prevents VRAM OOM errors on high-res frames
    tile_pad=10,
    pre_pad=0,
    half=True if torch.cuda.is_available() else False,
    device=device
)
print("Real-ESRGAN Loaded Successfully!")


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
    job_id = job_input.get("job_id", "job")

    work_dir = Path(f"/tmp/{job_id}")
    work_dir.mkdir(parents=True, exist_ok=True)

    input_video = work_dir / "input.mp4"
    output_video = work_dir / "output.mp4"
    frames_in_dir = work_dir / "frames_in"
    frames_out_dir = work_dir / "frames_out"

    frames_in_dir.mkdir(exist_ok=True)
    frames_out_dir.mkdir(exist_ok=True)

    try:
        # 1. Download video chunk
        res = requests.get(video_url, stream=True)
        with open(input_video, "wb") as f:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)

        # 2. Extract frames using FFmpeg
        subprocess.run([
            "ffmpeg", "-y", "-i", str(input_video),
            "-q:v", "1", str(frames_in_dir / "frame_%08d.jpg")
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 3. Upscale each frame using Real-ESRGAN in PyTorch
        in_frames = sorted(list(frames_in_dir.glob("*.jpg")))
        for frame_path in in_frames:
            img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            output_img, _ = upsampler.enhance(img, outscale=scale)
            cv2.imwrite(str(frames_out_dir / frame_path.name), output_img)

        # 4. Reassemble upscaled frames into video
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(frames_out_dir / "frame_%08d.jpg"),
            "-i", str(input_video),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy",
            str(output_video)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 5. Upload upscaled chunk to Nebius
        output_url = upload_to_s3(str(output_video), f"{job_id}_upscaled.mp4")
        return {"upscaled_video_url": output_url}

    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir)


async def async_handler(job):
    job_input = job["input"]
    job_input["job_id"] = job.get("id", "job")
    
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, process_video_sync, job_input)
    return result


runpod.serverless.start({"handler": async_handler})