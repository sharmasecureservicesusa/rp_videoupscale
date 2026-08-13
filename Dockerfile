FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# 1. Install system dependencies & fix cleanup path
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# 3. Install Python dependencies (EXCLUDING torchvision to preserve base CUDA bindings)
RUN pip install --no-cache-dir \
    opencv-python-headless \
    boto3 \
    python-dotenv \
    requests \
    runpod \
    gdown \
    scipy \
    pyyaml

# 4. Install BasicSR and RealESRGAN without dependencies
RUN pip install --no-cache-dir --no-deps basicsr
RUN pip install --no-cache-dir --no-deps realesrgan

# 5. Lock NumPy to 1.26.4 to avoid NumPy 2.x C-API breaking changes with PyTorch 2.1
RUN pip uninstall -y numpy && pip install --force-reinstall --no-cache-dir "numpy==1.26.4"

# 6. Verify PyTorch CUDA availability and NumPy version during build
RUN python -c "import torch, numpy; print('=== CUDA AVAILABLE:', torch.cuda.is_available(), '| NUMPY VERSION:', numpy.__version__, '===')"

# 7. Download RealESRGAN_x4plus weights into image layer for instant warm starts
RUN mkdir -p /app/weights && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O /app/weights/RealESRGAN_x4plus.pth

# 8. Copy worker script
COPY async_handler.py /app/async_handler.py

CMD ["python", "-u", "/app/async_handler.py"]