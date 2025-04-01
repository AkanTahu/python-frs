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
    pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -r requirements.txt

COPY . .


ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]

EXPOSE 5000
