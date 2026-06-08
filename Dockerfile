FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download dense models at build time for faster cold starts.
# Each download is its own RUN so adding/removing one doesn't invalidate
# unrelated layer caches.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy ONLY the files needed to build indexes. This isolates the slow
# indexing layers from edits to search.py / eval.py / bench.py /
# retrievers / tests, so the dense-encoding steps stay cached across
# most source changes. If any of these specific files change, the
# affected index layers rebuild.
COPY src/__init__.py src/data.py src/build_index.py src/
COPY src/index/ src/index/

# Build BM25 + MiniLM dense indexes. `build_index.py` is idempotent at
# the artifact level, so re-running it does nothing if the indexes
# already exist.
RUN python src/build_index.py --models minilm

# Copy the rest of the source tree (retrievers, eval, bench, tests,
# Makefile, docs). Edits to these will NOT invalidate the indexing
# layers above.
COPY . .

# Default: run the full benchmark using the baked-in indexes. Mount a
# volume at /app/results to capture results/bench.json on the host.
CMD ["make", "bench"]
