FROM python:3.9.9-slim
LABEL org.opencontainers.image.source https://github.com/serengil/deepface

WORKDIR /app

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

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN mkdir -p dataset
RUN mkdir -p result

RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -r /app/requirements.txt

# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -e .

ENV PYTHONUNBUFFERED=1

EXPOSE 5000