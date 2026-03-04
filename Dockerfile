FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    cmake \
    rsync \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    gradio[oauth]==4.0.0 \
    "uvicorn>=0.14.0" "websockets>=10.4" \
    spaces "fastapi<0.113.0"

# Copy application files
COPY . .

# Create user directory
RUN mkdir -p /home/user && ln -s /app /home/user/app

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
