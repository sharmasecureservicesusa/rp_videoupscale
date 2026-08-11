FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone ComfyUI and custom nodes
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI
WORKDIR /workspace/ComfyUI

RUN pip install --no-cache-dir -r requirements.txt
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git /workspace/ComfyUI/custom_nodes/ComfyUI-Manager
RUN git clone https://github.com/pydantic/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite

# Bake Model Weights directly into image
RUN mkdir -p /workspace/ComfyUI/models/upscale_models
RUN wget -O /workspace/ComfyUI/models/upscale_models/RealESRGAN_x4plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

EXPOSE 8188

CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]