FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PyTorch and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Using --extra-index-url for CPU-only PyTorch to reduce image size
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Create non-root user for security
RUN useradd -m -u 1000 birdsense && \
    mkdir -p /data /data/images && \
    chown -R birdsense:birdsense /app /data

# Copy application code
COPY --chown=birdsense:birdsense config.py bridge.py classifier.py database.py ./
COPY --chown=birdsense:birdsense bird_names_nl.json ./

USER birdsense

# Health check using API endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:5001/api/health || exit 1

# Run the bridge
CMD ["python", "-u", "bridge.py"]
