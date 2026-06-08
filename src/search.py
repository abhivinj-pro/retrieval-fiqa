"""CLI entry point for the retrieval system."""
import argparse
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.index.dense_index import MODEL_REGISTRY, DEFAULT_MODEL_KEY
from src.data import load_fiqa


def get_retriever(method: str, dense_model: str = DEFAULT_MODEL_KEY,
                  rrf_k: int = HybridRetriever.DEFAULT_RRF_K,
                  fetch_k: int = HybridRetriever.DEFAULT_FETCH_K,
                  weights=HybridRetriever.DEFAULT_WEIGHTS,
                  fusion: str = HybridRetriever.DEFAULT_FUSION):
    """Load the appropriate retriever."""
    if method == "bm25":
        return BM25Retriever()
    if method == "dense":
        return DenseRetriever(model_key=dense_model)
    if method == "hybrid":
        bm25 = BM25Retriever()
        dense = DenseRetriever(model_key=dense_model)
        return HybridRetriever(bm25, dense, rrf_k=rrf_k, fetch_k=fetch_k,
                               weights=weights, fusion=fusion)
    raise ValueError(f"Unknown method: {method}. Choose from: bm25, dense, hybrid")


def main():
    parser = argparse.ArgumentParser(description="Financial Q&A Retrieval System")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results (default: 10)")
    parser.add_argument("--method", type=str, default="hybrid",
                        choices=["bm25", "dense", "hybrid"],
                        help="Retrieval method (default: hybrid)")
    parser.add_argument("--dense-model", type=str, default=DEFAULT_MODEL_KEY,
                        choices=list(MODEL_REGISTRY.keys()),
                        help=f"Dense model (default: {DEFAULT_MODEL_KEY}). "
                             "Used for 'dense' and 'hybrid' methods.")
    parser.add_argument("--rrf-k", type=int, default=HybridRetriever.DEFAULT_RRF_K,
                        help=f"RRF k (default: {HybridRetriever.DEFAULT_RRF_K})")
    parser.add_argument("--fetch-k", type=int, default=HybridRetriever.DEFAULT_FETCH_K,
                        help=f"Sub-retriever fetch size for hybrid "
                             f"(default: {HybridRetriever.DEFAULT_FETCH_K})")
    parser.add_argument("--w-bm25", type=float,
                        default=HybridRetriever.DEFAULT_WEIGHTS[0],
                        help=f"Hybrid BM25 RRF weight (default: {HybridRetriever.DEFAULT_WEIGHTS[0]})")
    parser.add_argument("--w-dense", type=float,
                        default=HybridRetriever.DEFAULT_WEIGHTS[1],
                        help=f"Hybrid dense RRF weight (default: {HybridRetriever.DEFAULT_WEIGHTS[1]})")
    parser.add_argument("--fusion", type=str,
                        default=HybridRetriever.DEFAULT_FUSION,
                        choices=["rrf", "score"],
                        help=f"Hybrid fusion method (default: {HybridRetriever.DEFAULT_FUSION})")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    retriever = get_retriever(
        args.method,
        dense_model=args.dense_model,
        rrf_k=args.rrf_k,
        fetch_k=args.fetch_k,
        weights=(args.w_bm25, args.w_dense),
        fusion=args.fusion,
    )
    results = retriever.search(args.query, top_k=args.top_k)
    
    # Load corpus for passage text display
    corpus, _, _ = load_fiqa()
    
    if args.json:
        output = []
        for rank, (doc_id, score) in enumerate(results, 1):
            doc = corpus.get(doc_id, {})
            output.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": round(score, 6),
                "title": doc.get("title", ""),
                "text": doc.get("text", "")[:300]
            })
        print(json.dumps(output, indent=2))
    else:
        print(f"\nQuery: {args.query}")
        print(f"Method: {args.method} | Dense model: {args.dense_model} | Top-K: {args.top_k}")
        print("=" * 80)
        for rank, (doc_id, score) in enumerate(results, 1):
            doc = corpus.get(doc_id, {})
            title = doc.get("title", "")
            text = doc.get("text", "")[:200]
            print(f"\n[{rank}] Score: {score:.6f} | Doc ID: {doc_id}")
            if title:
                print(f"    Title: {title}")
            print(f"    {text}...")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
