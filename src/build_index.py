"""Build all indexes for the retrieval system.

Idempotent at the artifact level: each index is skipped if its sentinel
file already exists. Pass ``--force`` to rebuild from scratch.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_fiqa, get_corpus_texts
from src.index.bm25_index import INDEX_DIR as BM25_INDEX_DIR, build_bm25_index
from src.index.dense_index import (
    MODEL_REGISTRY,
    artifact_path_for,
    build_dense_index,
    resolve_model,
)


def _bm25_sentinel() -> str:
    return os.path.join(BM25_INDEX_DIR, "bm25", "bm25_index.pkl")


def _dense_sentinel(model_key: str) -> str:
    return os.path.join(artifact_path_for(model_key), "faiss.index")


def main():
    parser = argparse.ArgumentParser(description="Build retrieval indexes")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        help=(
            "Dense model keys to build. Choices: "
            f"{', '.join(MODEL_REGISTRY.keys())}. Default: build all. "
            "Each model is skipped if its index already exists."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Encoding batch size (default: 256)")
    parser.add_argument("--skip-bm25", action="store_true", help="Skip BM25 index build")
    parser.add_argument("--skip-dense", action="store_true", help="Skip all dense index builds")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if artifacts already exist")

    args = parser.parse_args()

    print("Loading FiQA corpus...")
    corpus, _, _ = load_fiqa()
    doc_ids, texts = get_corpus_texts(corpus)
    print(f"Loaded {len(doc_ids)} documents")

    # --- BM25 ---
    if not args.skip_bm25:
        if not args.force and os.path.exists(_bm25_sentinel()):
            print(f"\n[skip] BM25 index already exists at {_bm25_sentinel()}")
        else:
            print("\n--- Building BM25 Index (word tokenizer) ---")
            start = time.time()
            build_bm25_index(doc_ids, texts)
            print(f"BM25 index built in {time.time() - start:.1f}s")

    # --- Dense (one or more) ---
    if not args.skip_dense:
        for key in args.models:
            hf_name, _, _ = resolve_model(key)
            sentinel = _dense_sentinel(key)
            if not args.force and os.path.exists(sentinel):
                print(f"\n[skip] Dense index '{key}' ({hf_name}) already exists at {sentinel}")
                continue
            print(f"\n--- Building Dense Index '{key}' ({hf_name}) ---")
            start = time.time()
            build_dense_index(
                doc_ids, texts,
                model_name=hf_name,
                batch_size=args.batch_size,
                save_path=artifact_path_for(key),
            )
            print(f"Dense index '{key}' built in {time.time() - start:.1f}s")

    print("\nAll requested indexes are present.")
    print(f"Artifacts under: {os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'index_artifacts')}")


if __name__ == "__main__":
    main()
