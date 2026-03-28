FROM python:3.9-slim-bookworm
ENV PYTHONUNBUFFERED=1

ENV LIBGL_ALWAYS_INDIRECT=1
ENV MESA_LOADER_DRIVER_OVERRIDE=swrast
ENV QT_QPA_PLATFORM=offscreen

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxi6 libxrender1 libxext6 libxtst6 libxcursor1 libxfixes3 \
    libxrandr2 libxinerama1 libxkbcommon-x11-0 libxkbcommon0 libxkbfile1 \
    libxcb1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
    libxcb-render-util0 libxcb-shape0 libxcb-shm0 libxcb-xfixes0 \
    libxcb-xinerama0 libxcb-xkb1 libgl1 libglx0 libegl1 libgles2 \
    libglvnd0 libglib2.0-0 libsm6 libxau6 libxdmcp6 libxmu6 libxpm4 \
    libxxf86vm1 fontconfig libfreetype6 libpng16-16 libjpeg62-turbo \
    wget xz-utils ca-certificates \
    && rm -rf /var/lib/apt/lists/* && ldconfig

ARG BLENDER_VERSION=5.1.0
RUN wget -q https://download.blender.org/release/Blender5.1/blender-${BLENDER_VERSION}-linux-x64.tar.xz \
    && tar -xf blender-${BLENDER_VERSION}-linux-x64.tar.xz \
    && mv blender-${BLENDER_VERSION}-linux-x64 /opt/blender \
    && rm blender-${BLENDER_VERSION}-linux-x64.tar.xz \
    && /opt/blender/blender --version

ENV PATH="/opt/blender:${PATH}"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ml-service/app/ ./app/
COPY ml-service/blender_data/ ./blender_data/
COPY ml-service/app/models/ ./models/

RUN mkdir -p /app/uploads /app/results

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]