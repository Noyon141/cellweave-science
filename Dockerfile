# CellWeave Science Engine — Execution Container Image
# Authoritative Base: Python 3.11 Debian Slim

FROM python:3.11-slim AS runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged application user
RUN groupadd -g 10001 cellweave && \
    useradd -u 10001 -g cellweave -m -s /bin/bash cellweave

WORKDIR /app

# Copy dependency configuration and install
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir hatchling && \
    pip install --no-cache-dir .

# Copy application source code
COPY cellweave_science/ ./cellweave_science/
COPY modal_app/ ./modal_app/

# Switch to unprivileged user
USER cellweave:cellweave

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    CELLWEAVE_ENV=production

# Healthcheck entrypoint
CMD ["python", "-c", "import cellweave_science; print('CellWeave Science Engine ready:', cellweave_science.__version__)"]
