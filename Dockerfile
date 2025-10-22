# Use Ubuntu 24.04 as base image
FROM ubuntu:24.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Update system and install required packages
RUN apt update && apt install -y \
    python3.12 python3.12-venv python3.12-dev \
    curl wget git unzip net-tools \
    libgl1 libglib2.0-0 \
    && apt clean && rm -rf /var/lib/apt/lists/*

# Set Python3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Create a working directory
WORKDIR /app

# Create models directory structure
# RUN mkdir -p /app/models/yolov8n

# Copy application files (excluding empty directories)
COPY . /app

# Copy YOLOv8n model files to the correct location
# COPY models/yolov8n/yolov8n.xml /app/models/yolov8n/
# COPY models/yolov8n/yolov8n.bin /app/models/yolov8n/
# COPY models/yolov8n/model.json /app/models/yolov8n/

# Ensure models directory exists (for volume mounting)
# NOT monting model folder until we are ready to manage model from the applications
# VOLUME ["/app/models"]

# Create a virtual environment
RUN python3 -m venv /app/venv

# Activate virtual environment
ENV PATH="/app/venv/bin:$PATH"

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Expose FastAPI port
EXPOSE 8001

# Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
