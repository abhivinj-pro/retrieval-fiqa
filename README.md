# FiQA Financial Q&A Retrieval System

A production-constrained retrieval system over the BEIR/FiQA financial Q&A corpus, implementing BM25, Dense, and Hybrid retrieval strategies.

## Constraints

- **CPU only** — no GPU at index or serve time
- **Latency** — p95 ≤ 50 ms per query (after warmup)
- **Memory** — index footprint ≤ 2 GB RAM at serve time

## Quick Start

### Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download data & build indexes
make index

# Run evaluation
make eval

# Full benchmark with ablations
make bench

# Search
python src/search.py --query "what is short selling?" --top-k 10 --method hybrid
```

### Docker (Reproducible)

The image bakes in (a) the dense model `all-MiniLM-L6-v2`, (b) the FiQA
dataset, and (c) **pre-built BM25 + dense indexes**. Dense encoding of 57K
passages on CPU takes ~30-40 min inside Docker, so we move that cost to image
build time. The reviewer's `docker run` then jumps straight to the benchmark
(~30s).

```bash
# 1. Build the image (one-time, ~30-40 min — pre-builds the index).
docker build -t fiqa-retrieval .

# 2. Reproduce the full benchmark (~30 s).
#    Mount ./results so bench.json lands on your host.

# Linux / macOS:
docker run --rm -v "$(pwd)/results:/app/results" fiqa-retrieval

# Windows PowerShell:
docker run --rm -v "${PWD}/results:/app/results" fiqa-retrieval

# Windows CMD:
docker run --rm -v "%cd%/results:/app/results" fiqa-retrieval
```

Other entry points (override `CMD`):

```bash
# Run eval only (BM25 / dense / hybrid, no ablations)
docker run --rm -v "${PWD}/results:/app/results" fiqa-retrieval make eval

# Force-rebuild the index from scratch (useful to verify reproducibility)
docker run --rm fiqa-retrieval python src/build_index.py

# Interactive search
docker run --rm fiqa-retrieval \
    python src/search.py --query "what is short selling?" --top-k 10
```

### CLI Usage

```bash
python src/search.py --query "what is short selling?" --top-k 10
python src/search.py --query "how do dividends work?" --method bm25 --top-k 5
python src/search.py --query "tax loss harvesting" --method dense --json
```

## Project Structure

```
├── README.md             # This file
├── DESIGN.md             # Design doc with benchmarks & analysis
├── Dockerfile            # Reproducible runtime
├── Makefile              # make index, make eval, make bench
├── src/
│   ├── search.py         # CLI entry point
│   ├── build_index.py    # Index builder
│   ├── eval.py           # Evaluation harness
│   ├── bench.py          # Full benchmark runner
│   ├── data.py           # Data loading utilities
│   ├── index/            # Indexing code (BM25, dense)
│   └── retrieval/        # Retriever implementations
├── tests/                # Unit tests
├── results/
│   ├── bench.json        # Benchmark output
│   └── failures.md       # Failure case analysis
└── requirements.txt
```

## Retrieval Methods

| Method | Description |
|--------|-------------|
| `bm25` | BM25 Okapi over a byte-level **BPE** tokenizer trained on the FiQA corpus (preserves financial symbols like `$`, `%`, `&`, tickers, and decimal amounts) |
| `dense` | all-MiniLM-L6-v2 + FAISS flat index (cosine similarity) |
| `hybrid` | BM25 + Dense combined via Reciprocal Rank Fusion (RRF) |

## Testing

```bash
make test
# or
python -m pytest tests/ -v
```
