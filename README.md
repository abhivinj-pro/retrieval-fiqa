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

The image bakes in (a) the dense model `sentence-transformers/all-MiniLM-L6-v2`,
(b) the FiQA dataset, and (c) **pre-built BM25 + MiniLM dense indexes**.
Dense encoding of 57K passages on CPU takes ~30-40 min inside Docker, so
moved that cost to image build time. The `docker run` then
jumps straight to the benchmark.

The image is self-contained: dataset, model, and indexes are baked in at
build time. **Do not bind-mount `./index_artifacts` or `./datasets` from
the host** — empty/stale host directories will shadow the baked-in files
and break the run. Only mount `./results` if you want `bench.json` on
your host.

```bash
# 1. Build the image (one-time — pre-builds all indexes).
docker build -t fiqa-retrieval .

# 2. Reproduce the full benchmark (default CMD runs `make bench`).
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
# Run eval only (BM25 / dense / hybrid)
docker run --rm -v "${PWD}/results:/app/results" fiqa-retrieval make eval

# Generate results/failures.md (and failures_raw.json) from the hybrid retriever
docker run --rm -v "${PWD}/results:/app/results" fiqa-retrieval python src/analyze_failures.py

# Interactive search (hybrid, default)
docker run --rm fiqa-retrieval python src/search.py --query "what is short selling?" --top-k 10

# Run unit tests
docker run --rm fiqa-retrieval make test

# Force-rebuild the dense index from scratch
docker run --rm fiqa-retrieval python src/build_index.py --skip-bm25 --force
```

### CLI Usage

```bash
python src/search.py --query "what is short selling?" --top-k 10
python src/search.py --query "how do dividends work?" --method bm25 --top-k 5
python src/search.py --query "tax loss harvesting" --method dense --json
python src/search.py --query "what is short selling?" --method hybrid --w-bm25 0.3 --w-dense 0.7
```

## Project Structure

```
├── README.md             # This file
├── DESIGN.md             # Design doc with benchmarks & analysis
├── Dockerfile            # Reproducible runtime
├── Makefile              # make index, make eval, make bench
├── src/
│   ├── search.py             # CLI entry point
│   ├── build_index.py        # Index builder
│   ├── eval.py               # Evaluation harness
│   ├── bench.py              # Full benchmark runner
│   ├── analyze_failures.py   # Generates results/failures.md
│   ├── data.py               # Data loading utilities
│   ├── index/                # Indexing code (BM25, dense)
│   └── retrieval/            # Retriever implementations
├── tests/                    # Unit tests
├── results/
│   ├── bench.json            # Benchmark output
│   └── failures.md           # Failure case analysis
└── requirements.txt
```

## Retrieval Methods

| Method | Description |
|--------|-------------|
| `bm25` | BM25 Okapi over a Lucene-StandardAnalyzer-style English tokenizer (NFKC + lowercase, finance-symbol preservation for `$1.2B` / `2.5%` / `10-K` / `s&p`, English stopword removal, Porter stemming) |
| `dense` | `sentence-transformers/all-MiniLM-L6-v2` (384d, ~80MB) encoder + FAISS flat index (cosine). The encoder is pluggable via `MODEL_REGISTRY` in `src/index/dense_index.py`; `minilm` is the only model currently shipped. |
| `hybrid` | BM25 + Dense combined via **weighted** Reciprocal Rank Fusion (RRF). Default weights `(w_bm25=0.3, w_dense=0.7)` bias toward the stronger dense retriever on this corpus so that Hybrid ≥ Dense > BM25. |

## Testing

```bash
make test
# or
python -m pytest tests/ -v
```
