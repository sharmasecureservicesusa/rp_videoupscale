FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt-get/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install runtime dependencies
RUN pip install --no-cache-dir opencv-python-headless boto3 python-dotenv requests runpod gdown torchvision

# Install BasicSR and RealESRGAN without dependencies (prevents pip from upgrading numpy/torch)
RUN pip install --no-cache-dir --no-deps basicsr
RUN pip install --no-cache-dir --no-deps realesrgan

# FORCE uninstall any existing numpy and lock to 1.26.4
RUN pip uninstall -y numpy && pip install --force-reinstall --no-cache-dir "numpy==1.26.4"

# BUILD-TIME VERIFICATION: Ensure NumPy 1.26.4 is installed correctly
RUN python -c "import numpy; print('=== INSTALLED NUMPY VERSION:', numpy.__version__, '===')"

# Copy model weights and handler
COPY weights /app/weights
COPY async_handler.py /app/async_handler.py

CMD ["python", "-u", "/app/async_handler.py"]