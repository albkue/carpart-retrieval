# ML Search Service Dockerfile
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system dependencies for OpenCV and ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgflags-dev \
    libsnappy-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Increase pip timeout and retries to handle flaky PyPI connections
ENV PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=5

# 1. Install CPU-only torch first (separate index URL, ~200MB vs ~2.4GB for CUDA)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.0

# 3. Copy and install remaining app dependencies (excludes already-installed heavy deps)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data
RUN mkdir -p /app/data/faiss_index

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8001/health')" || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
