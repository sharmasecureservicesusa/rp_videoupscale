import os
import json
import shutil
import subprocess
import requests
import asyncio
import boto3
import torch
import cv2
import numpy as np
import runpod
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Process GPU tasks sequentially per worker instance
executor = ThreadPoolExecutor(max_workers=1)

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


def get_video_info(video_path: Path):
    """Extracts width, height, and FPS directly using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json",
        str(video_path)
    ]
    res = subprocess.check_output(cmd).decode()
    info = json.loads(res)["streams"][0]
    w, h = int(info["width"]), int(info["height"])

    fps_str = info.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = map(float, fps_str.split("/"))
        fps = num / den if den != 0 else 30.0
    else:
        fps = float(fps_str)

    return w, h, str(fps)


def upload_to_s3(local_path: str, filename: str, job_input: dict) -> str:
    """Uploads upscaled chunk back to Nebius S3 using payload credentials or env vars."""
    bucket = job_input.get("s3_bucket") or os.getenv("S3_BUCKET")
    endpoint_url = job_input.get("s3_endpoint_url") or os.getenv("S3_ENDPOINT_URL", "https://storage.eu-north1.nebius.cloud:443")
    key_id = job_input.get("aws_access_key_id") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = job_input.get("aws_secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY")
    region = job_input.get("aws_region") or os.getenv("AWS_REGION", "eu-north1")

    if not bucket:
        raise ValueError("S3_BUCKET was not provided in job payload or environment variables.")

    client = boto3.client(
        "s3",
        aws_access_key_id=key_id,
        aws_secret_access_key=secret_key,
        region_name=region,
        endpoint_url=endpoint_url
    )

    s3_key = f"outputs/{filename}"
    client.upload_file(local_path, bucket, s3_key)
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": s3_key},
        ExpiresIn=3600
    )


def process_video_sync(job_input: dict) -> dict:
    video_url = job_input.get("video_url")
    requested_scale = job_input.get("scale", 4)
    job_id = job_input.get("job_id", "job")

    work_dir = Path(f"/tmp/{job_id}")
    work_dir.mkdir(parents=True, exist_ok=True)

    input_video = work_dir / "input.mp4"
    output_video = work_dir / "output.mp4"

    try:
        # 1. Download input chunk
        res = requests.get(video_url, stream=True)
        with open(input_video, "wb") as f:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)

        # 2. Extract input dimensions and FPS
        in_w, in_h, fps = get_video_info(input_video)

        # 3. Calculate effective scale capped at 4K (3840x2160)
        max_w, max_h = 3840, 2160
        max_allowed_scale = min(max_w / in_w, max_h / in_h)
        effective_scale = min(float(requested_scale), max_allowed_scale)
        effective_scale = max(1.0, effective_scale)

        out_w = int(in_w * effective_scale)
        out_h = int(in_h * effective_scale)

        # Enforce even dimensions required for H.264 video streams
        out_w = out_w if out_w % 2 == 0 else out_w - 1
        out_h = out_h if out_h % 2 == 0 else out_h - 1

        frame_size = in_w * in_h * 3

        # 4. Start FFmpeg pipe reader (Decodes video frames directly into RAM)
        reader = subprocess.Popen([
            "ffmpeg", "-v", "error",
            "-i", str(input_video),
            "-f", "image2pipe", "-pix_fmt", "bgr24", "-vcodec", "rawvideo", "-"
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        # 5. Start FFmpeg pipe writer (Encodes upscaled frames & copies audio stream directly)
        writer = subprocess.Popen([
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{out_w}x{out_h}",
            "-pix_fmt", "bgr24",
            "-r", fps,
            "-i", "-",
            "-i", str(input_video),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "faster",
            "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy",
            str(output_video)
        ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        # 6. Stream frames through Real-ESRGAN in VRAM without touching disk
        while True:
            raw_frame = reader.stdout.read(frame_size)
            if not raw_frame or len(raw_frame) != frame_size:
                break

            img = np.frombuffer(raw_frame, dtype=np.uint8).reshape((in_h, in_w, 3))

            # Upscale frame in GPU VRAM
            output_img, _ = upsampler.enhance(img, outscale=effective_scale)

            # Ensure image dimensions match target frame size exactly
            if output_img.shape[1] != out_w or output_img.shape[0] != out_h:
                output_img = cv2.resize(output_img, (out_w, out_h), interpolation=cv2.INTER_AREA)

            # Write frame to FFmpeg encoder stream
            writer.stdin.write(output_img.tobytes())

        # Close process streams
        reader.stdout.close()
        reader.wait()
        writer.stdin.close()
        writer.wait()

        # 7. Upload upscaled chunk back to Nebius S3
        output_url = upload_to_s3(str(output_video), f"{job_id}_upscaled.mp4", job_input)
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