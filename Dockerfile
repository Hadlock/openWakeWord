# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    OPENWAKEWORD_WORKDIR=/workspace \
    OPENWAKEWORD_SKIP_TFLITE=1 \
    OPENWAKEWORD_PIPER_DIR=/opt/piper-sample-generator \
    OPENWAKEWORD_PIPER_MODEL=en_US-libritts_r-medium.pt \
    OPENWAKEWORD_PIPER_MODEL_URL="https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt" \
    OPENWAKEWORD_RESOURCES_DIR=/app/openwakeword/resources/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        unzip \
        libsndfile1 \
        ffmpeg \
        espeak-ng \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

# Install dependencies in a stable order to avoid numpy conflicts between torch and tensorflow.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy==1.22.4 && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1 \
        torchaudio==2.4.1 && \
    pip install --no-cache-dir \
        onnxruntime>=1.19.0 \
        webrtcvad \
        piper-phonemize==1.1.0 \
        mutagen==1.47.0 \
        torchinfo==1.8.0 \
        torchmetrics==1.2.0 \
        speechbrain==0.5.14 \
        audiomentations==0.33.0 \
        torch-audiomentations==0.11.0 \
        acoustics==0.2.6 \
        pyarrow==17.0.0 \
        onnx==1.14.0 \
        pronouncing==0.2.0 \
        datasets==2.14.6 \
        deep-phonemizer==0.0.19 && \
    pip install --no-cache-dir -e .

# Prepare Piper sample generator used during synthetic data generation.
RUN git clone https://github.com/rhasspy/piper-sample-generator ${OPENWAKEWORD_PIPER_DIR} && \
    mkdir -p ${OPENWAKEWORD_PIPER_DIR}/models && \
    wget -O ${OPENWAKEWORD_PIPER_DIR}/models/${OPENWAKEWORD_PIPER_MODEL} ${OPENWAKEWORD_PIPER_MODEL_URL}

# Fetch required openWakeWord resource models for ONNX inference.
RUN mkdir -p ${OPENWAKEWORD_RESOURCES_DIR} && \
    wget -O ${OPENWAKEWORD_RESOURCES_DIR}/embedding_model.onnx https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx && \
    wget -O ${OPENWAKEWORD_RESOURCES_DIR}/melspectrogram.onnx https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx

EXPOSE 8080

ENTRYPOINT ["python", "-m", "docker_entrypoint"]
