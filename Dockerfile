FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv --no-cache-dir

# Copy dependency files first for layer caching
# README.md is required by hatchling during the editable install
COPY pyproject.toml uv.lock README.md ./

# Install all dependencies (kbase-core is vendored — no git fetch needed)
RUN uv pip install --system -e . --no-cache-dir

# Copy application source and vendored libraries
COPY src/ ./src/
COPY vendor/ ./vendor/
COPY main.py ./

# Default to HTTP transport for Railway
ENV TRANSPORT=http

# Railway injects PORT at runtime
CMD ["python", "main.py"]
