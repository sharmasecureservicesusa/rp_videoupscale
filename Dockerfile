FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir "numpy<2.0.0"
RUN pip install --no-cache-dir \
    runpod \
    requests \
    realesrgan \
    torchvision

# Bake upscale weights into image layer
RUN mkdir -p /app/weights && \
    wget -O /app/weights/RealESRGAN_x4plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

COPY async_handler.py /app/async_handler.py

CMD ["python", "-u", "/app/async_handler.py"]