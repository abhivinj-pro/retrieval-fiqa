FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Download model at build time for faster cold starts
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Pre-build indexes at image-build time. Dense encoding of 57K FiQA passages
# on CPU is slow (~30-40 min inside Docker on Windows/WSL2), so we bake the
# artifacts into the image. The reviewer's `docker run` then jumps straight to
# `make bench` (~30s) instead of re-encoding. The regeneration script
# (src/build_index.py) is preserved and can be re-invoked manually:
#   docker run --rm fiqa-retrieval python src/build_index.py
RUN python src/build_index.py

# Default: run the benchmark using the baked-in indexes. Mount a volume at
# /app/results to capture results/bench.json on the host.
CMD ["make", "bench"]
