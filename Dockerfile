# Use Python 3.11
FROM python:3.11-slim
 
# Keep python output unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
 
WORKDIR /app
 
# Install system dependencies + curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential git netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*
 
# Install uv using official method
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:/root/.local/bin:$PATH"
 
# Copy ALL files first (simpler approach)
COPY . /app
 
# Install dependencies using pip (fallback - most reliable)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -e .
 
# Add convenient CLI wrappers for train and inference (use the class-based pipeline in `src`)
RUN printf '#!/usr/bin/env bash\npython -m scripts.pipeline "$@"' > /usr/local/bin/train && \
    chmod +x /usr/local/bin/train && \
    printf '#!/usr/bin/env bash\npython -m scripts.pipeline "$@"' > /usr/local/bin/inference && \
    chmod +x /usr/local/bin/inference
 
# Expose MLflow UI port (optional)
EXPOSE 5000
 
# Default command: keep container running
CMD ["bash", "-lc", "tail -f /dev/null"]