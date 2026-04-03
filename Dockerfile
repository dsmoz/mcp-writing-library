FROM python:3.11-slim

WORKDIR /app

# git is required to install kbase-core from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv --no-cache-dir

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install all dependencies (kbase-core will be fetched from GitHub)
RUN uv pip install --system -e . --no-cache-dir

# Copy application source
COPY src/ ./src/
COPY main.py ./

# Default to HTTP transport for Railway
ENV TRANSPORT=http

# Railway injects PORT at runtime
CMD ["python", "main.py"]
