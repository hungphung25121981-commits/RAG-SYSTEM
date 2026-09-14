FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive\
	PYTHONUNBUFFERED=1\
	PYTHONDONTWRITEBYTECODE=12

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsndfile1 \
    build-essential \
    git \
    curl \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*
# Cài bản wheel dựng sẵn cho CUDA 12.1 (Tải 5 giây là xong, đéo lo thiếu Ninja/CMake)
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
ENV FORCE_CMAKE=1

RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

WORKDIR /app/backend 


COPY ./backend/requirements.txt ./

RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --default-timeout=1000 --retries 10 \
    onnxruntime-gpu rapidocr-onnxruntime easyocr ultralytics kuzu trulens-eval

# 2. Cài toàn bộ các gói còn lại trong requirements.txt
RUN pip install --default-timeout=1000 --retries 10 -r requirements.txt

COPY ./backend/ .
RUN pip3 install --no-cache-dir flash-attn>=2.5.0 --no-build-isolation || true
ENV PYTHONPATH="/app/backend/src:${PYTHONPATH}"


CMD ["python3", "backend/cli_pipeline.py","--help"]