FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies (including wget for model weights)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    curl \
    && rm -rf /var/lib/apt-get/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install runtime Python dependencies
RUN pip install --no-cache-dir \
    opencv-python-headless \
    boto3 \
    python-dotenv \
    requests \
    runpod \
    gdown \
    torchvision

# Install BasicSR and RealESRGAN without dependencies to prevent package conflicts
RUN pip install --no-cache-dir --no-deps basicsr
RUN pip install --no-cache-dir --no-deps realesrgan

# Force-reinstall and pin NumPy to 1.26.4 (fixes PyTorch C-API incompatibility)
RUN pip uninstall -y numpy && pip install --force-reinstall --no-cache-dir "numpy==1.26.4"

# Build-time verification: Confirm NumPy version
RUN python -c "import numpy; print('=== INSTALLED NUMPY VERSION:', numpy.__version__, '===')"

# Create weights directory and download RealESRGAN_x4plus.pth directly
RUN mkdir -p /app/weights && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O /app/weights/RealESRGAN_x4plus.pth

# Copy application script
COPY async_handler.py /app/async_handler.py

CMD ["python", "-u", "/app/async_handler.py"]