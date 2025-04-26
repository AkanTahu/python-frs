FROM python:3.9.9-slim
LABEL org.opencontainers.image.source https://github.com/serengil/deepface

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libhdf5-dev \
    build-essential \
    cmake \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend-simple-FRS

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "-w", "4", "--timeout", "300", "-b", "0.0.0.0:5000", "app:app"]


EXPOSE 5000
