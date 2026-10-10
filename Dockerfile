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

# Copy locked dependencies, vendor packages, and packaging metadata
COPY requirements-lock.txt pyproject.toml README.md ./
COPY vendor/ ./vendor/

# Install strictly from reproducible locked dependency set
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir hatchling && \
    pip install --no-cache-dir -r requirements-lock.txt && \
    pip install --no-cache-dir --no-deps . && \
    python -c "import site; p=site.getsitepackages()[0]+'/jaxlib_compat.pth'; open(p, 'w').write('import sys; from jax._src.lib import xla_client; sys.modules.setdefault(\"jaxlib.xla_extension\", xla_client)\n')"

# Copy application source code
COPY cellweave_science/ ./cellweave_science/
COPY modal_app/ ./modal_app/
COPY benchmarks/ ./benchmarks/

# Switch to unprivileged user
USER cellweave:cellweave

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    CELLWEAVE_ENV=production

# Healthcheck entrypoint
CMD ["python", "-c", "import cellweave_science; print('CellWeave Science Engine ready:', cellweave_science.__version__)"]
