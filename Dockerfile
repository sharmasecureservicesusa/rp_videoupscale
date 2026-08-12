FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies (--no-deps on basicsr prevents it from overriding PyTorch/torchvision)
RUN pip install --no-cache-dir opencv-python-headless boto3 python-dotenv requests runpod
RUN pip install --no-cache-dir --no-deps basicsr
RUN pip install --no-cache-dir realesrgan

# FORCE NumPy 1.26.4 at the end so no package upgrades it to NumPy 2.x
RUN pip install --no-cache-dir "numpy==1.26.4"

# Copy weights and handler
COPY weights /app/weights
COPY async_handler.py /app/async_handler.py

CMD ["python", "-u", "/app/async_handler.py"]