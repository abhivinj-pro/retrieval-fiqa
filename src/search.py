"""CLI entry point for the retrieval system."""
import argparse
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.data import load_fiqa, get_corpus_texts


def get_retriever(method: str, rrf_k: int = 60, fetch_k: int = 100):
    """Load the appropriate retriever."""
    if method == "bm25":
        return BM25Retriever()
    elif method == "dense":
        return DenseRetriever()
    elif method == "hybrid":
        bm25 = BM25Retriever()
        dense = DenseRetriever()
        return HybridRetriever(bm25, dense, rrf_k=rrf_k, fetch_k=fetch_k)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: bm25, dense, hybrid")


def main():
    parser = argparse.ArgumentParser(description="Financial Q&A Retrieval System")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results (default: 10)")
    parser.add_argument("--method", type=str, default="hybrid",
                        choices=["bm25", "dense", "hybrid"],
                        help="Retrieval method (default: hybrid)")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF k parameter (default: 60)")
    parser.add_argument("--fetch-k", type=int, default=100,
                        help="Sub-retriever fetch size for hybrid (default: 100)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    retriever = get_retriever(args.method, rrf_k=args.rrf_k, fetch_k=args.fetch_k)
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
        print(f"Method: {args.method} | Top-K: {args.top_k}")
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
