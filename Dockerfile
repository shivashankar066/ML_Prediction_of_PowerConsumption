# ---------------------------
# 1. Base Image
# ---------------------------
FROM python:3.10-slim

# ---------------------------
# 2. Set Working Directory
# ---------------------------
WORKDIR /app

# ---------------------------
# 3. Copy project files
# ---------------------------
COPY . /app

# ---------------------------
# 4. Install system dependencies
# ---------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# 5. Upgrade pip & Install Python packages
# ---------------------------
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ---------------------------
# 6. Create directories (if not exist)
# ---------------------------
RUN mkdir -p data outputs logs

# ---------------------------
# 7. Default command
# ---------------------------
CMD ["python", "-m", "scripts.main"]
