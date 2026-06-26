FROM python:3.9-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LIBGL_ALWAYS_INDIRECT=1 \
    MESA_LOADER_DRIVER_OVERRIDE=swrast \
    QT_QPA_PLATFORM=offscreen \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        wget xz-utils ca-certificates \
        libgl1 libglx0 libegl1 libgles2 libglvnd0 libglib2.0-0 \
        libsm6 libxau6 libxdmcp6 libxmu6 libxpm4 libxxf86vm1 \
        libxi6 libxrender1 libxext6 libxtst6 libxcursor1 libxfixes3 \
        libxrandr2 libxinerama1 libxkbcommon-x11-0 libxkbcommon0 libxkbfile1 \
        libxcb1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
        libxcb-render-util0 libxcb-shape0 libxcb-shm0 libxcb-xfixes0 \
        libxcb-xinerama0 libxcb-xkb1 \
        fontconfig libfreetype6 libpng16-16 libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/* && ldconfig

ARG BLENDER_VERSION=5.1.0
RUN wget -q https://download.blender.org/release/Blender5.1/blender-${BLENDER_VERSION}-linux-x64.tar.xz \
    && tar -xf blender-${BLENDER_VERSION}-linux-x64.tar.xz \
    && mv blender-${BLENDER_VERSION}-linux-x64 /opt/blender \
    && rm blender-${BLENDER_VERSION}-linux-x64.tar.xz
ENV PATH="/opt/blender:${PATH}"

COPY mediapipe-0.8.10.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl /tmp/

RUN pip install --upgrade pip && pip install \
        fastapi==0.115.0 \
        uvicorn==0.30.6 \
        starlette==0.38.5 \
        sse-starlette==1.8.2 \
        "prometheus-fastapi-instrumentator>=0.9.1" \
        "prometheus_client>=0.20.0" \
        "celery[redis]==5.3.6" \
        redis==5.0.1 \
        pydantic==2.8.2 \
        pydantic_core==2.20.1 \
        pydantic-settings==2.4.0 \
        python-dotenv==1.0.1 \
        boto3==1.34.150 \
        botocore==1.34.150 \
        httpx==0.27.0 \
        "aiohttp>=3.9.0" \
        aiohttp-socks \
        numpy==1.24.4 \
        scipy==1.10.1 \
        opencv-contrib-python==4.9.0.80 \
        matplotlib==3.8.4 \
        Pillow==10.3.0 \
        PyGLM==2.5.7 \
        dtaidistance==2.3.10 \
        protobuf==4.25.9 \
        torch==2.1.2 \
        transformers==4.41.0 \
        accelerate==0.30.0 \
        sentence-transformers==2.7.0 \
        scikit-learn==1.3.2 \
        "ultralytics>=8.0.0" \
        "nudenet>=3.4.0" \
        onnxruntime \
        "aiogram>=3.7.0" \
        "yt-dlp>=2025.3.31" \
        "pytubefix>=8.0.0" \
    && pip install /tmp/mediapipe-0.8.10.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    && rm /tmp/mediapipe-0.8.10.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

RUN python - <<'EOF'
from ultralytics import YOLO
YOLO("yolov8n.pt")          # downloads and caches the weights
from nudenet import NudeDetector
NudeDetector()               # downloads and caches the ONNX model
print("Moderation models pre-downloaded OK")
EOF

COPY ml-service/app/        ./app/
COPY ml-service/blender_data/ ./blender_data/
COPY ml-service/app/models/ ./models/

RUN mkdir -p /app/uploads /app/results

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
