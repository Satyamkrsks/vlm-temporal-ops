# ─────────────────────────────────────────────────────────────────
# Base: CUDA 12.1 + Python 3.11 + Ubuntu 22.04
# ─────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# ─────────────────────────────────────────────────────────────────
# Install PyTorch with CUDA 12.1 wheels FIRST
# ─────────────────────────────────────────────────────────────────
RUN pip install \
    torch==2.2.2 \
    torchvision==0.17.2 \
    torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121

# ─────────────────────────────────────────────────────────────────
# Install Python dependencies
# ─────────────────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Install qwen_vl_utils
RUN pip install qwen-vl-utils || \
    pip install git+https://github.com/QwenLM/Qwen-VL.git#subdirectory=qwen_vl_utils

# ─────────────────────────────────────────────────────────────────
# Copy application code
# ─────────────────────────────────────────────────────────────────
COPY app/ /app/

# Model cache directory
RUN mkdir -p /root/.cache/huggingface/hub

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]