.PHONY: index eval bench search test clean

# Sentinel files produced by src/build_index.py. Using real files (rather
# than a .PHONY index target) lets `eval` / `bench` skip rebuilding when
# artifacts already exist, while still triggering a build on a fresh
# container. `build_index.py` is itself idempotent per-model, so the
# sentinels and the script agree on what counts as "already built".
BM25_INDEX      := index_artifacts/bm25/bm25_index.pkl
DENSE_MINILM    := index_artifacts/dense/faiss.index

$(BM25_INDEX) $(DENSE_MINILM):
	python src/build_index.py

# Build all indexes (BM25 + MiniLM). Idempotent: skips per-model
# if the artifact already exists.
index: $(BM25_INDEX) $(DENSE_MINILM)

# Run evaluation on all methods (auto-builds indexes if missing).
eval: index
	python src/eval.py --methods bm25 dense hybrid

# Run full benchmark suite with ablations (auto-builds indexes if missing).
bench: index
	python src/bench.py

# Interactive search (usage: make search QUERY="what is short selling?" METHOD=hybrid)
QUERY  ?= "what is short selling?"
METHOD ?= hybrid
TOP_K  ?= 10
MODEL  ?= minilm
search:
	python src/search.py --query $(QUERY) --method $(METHOD) --dense-model $(MODEL) --top-k $(TOP_K)

# Run unit tests
test:
	python -m pytest tests/ -v

# Clean artifacts
clean:
	rm -rf index_artifacts/ datasets/ __pycache__ src/__pycache__
	find . -name "*.pyc" -delete
