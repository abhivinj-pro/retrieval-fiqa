"""Evaluation harness for retrieval system benchmarking."""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_fiqa, get_corpus_texts
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


def recall_at_k(results: List[Tuple[str, float]], relevant: set, k: int = 10) -> float:
    """Compute Recall@K for a single query.
    
    Args:
        results: ranked list of (doc_id, score)
        relevant: set of relevant doc_ids
        k: cutoff
    
    Returns:
        Fraction of relevant docs found in top-k
    """
    if not relevant:
        return 0.0
    retrieved_ids = {doc_id for doc_id, _ in results[:k]}
    hits = len(retrieved_ids & relevant)
    return hits / len(relevant)


def mrr(results: List[Tuple[str, float]], relevant: set) -> float:
    """Compute Mean Reciprocal Rank for a single query.
    
    Args:
        results: ranked list of (doc_id, score)
        relevant: set of relevant doc_ids
    
    Returns:
        1/rank of first relevant result, or 0 if none found
    """
    for rank, (doc_id, _) in enumerate(results, 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def count_tokens(text: str) -> int:
    """Simple whitespace token count."""
    return len(text.split())


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def evaluate_retriever(retriever, queries: Dict, qrels: Dict, corpus: Dict,
                       top_k: int = 10, warmup_queries: int = 100,
                       label: str = "retriever") -> Dict:
    """Run full evaluation on a retriever.
    
    Args:
        retriever: retriever with .search(query, top_k) method
        queries: {query_id: query_text}
        qrels: {query_id: {doc_id: relevance}}
        corpus: {doc_id: {title, text}}
        top_k: evaluation cutoff
        warmup_queries: number of queries for warmup phase
        label: name for this configuration
    
    Returns:
        Dictionary with all evaluation metrics
    """
    query_ids = list(queries.keys())
    
    # Compute doc lengths for stratification
    doc_lengths = {}
    for doc_id, doc in corpus.items():
        text = doc.get("text", "") + " " + doc.get("title", "")
        doc_lengths[doc_id] = count_tokens(text)
    
    # Find top 10% longest docs
    all_lengths = sorted(doc_lengths.values(), reverse=True)
    length_threshold = all_lengths[int(len(all_lengths) * 0.1)] if all_lengths else 0
    long_doc_ids = {did for did, l in doc_lengths.items() if l >= length_threshold}
    
    # Memory before queries
    mem_before = get_memory_mb()
    
    # Cold-start latency (first 20 queries)
    cold_latencies = []
    cold_results = {}
    for qid in query_ids[:20]:
        start = time.perf_counter()
        results = retriever.search(queries[qid], top_k=top_k)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        cold_latencies.append(elapsed)
        cold_results[qid] = results
    
    # Warmup phase
    warmup_qids = query_ids[:warmup_queries] if len(query_ids) >= warmup_queries else query_ids
    for qid in warmup_qids[20:]:  # skip already-run cold queries
        retriever.search(queries[qid], top_k=top_k)
    
    # Warm evaluation (all queries)
    warm_latencies = []
    all_results = {}
    all_results.update(cold_results)  # keep cold results
    
    for qid in query_ids[20:]:  # run remaining queries
        start = time.perf_counter()
        results = retriever.search(queries[qid], top_k=top_k)
        elapsed = (time.perf_counter() - start) * 1000
        warm_latencies.append(elapsed)
        all_results[qid] = results
    
    # Also time the first 20 again for warm measurement
    for qid in query_ids[:20]:
        start = time.perf_counter()
        results = retriever.search(queries[qid], top_k=top_k)
        elapsed = (time.perf_counter() - start) * 1000
        warm_latencies.append(elapsed)
        all_results[qid] = results  # update with warm results
    
    mem_after = get_memory_mb()
    
    # Compute metrics
    recalls = []
    mrrs = []
    
    # Stratified containers
    short_recalls = []   # < 5 tokens
    medium_recalls = []  # 5-15 tokens
    long_recalls = []    # > 15 tokens
    
    long_doc_recalls = []   # gold passage in top 10% longest
    other_doc_recalls = []  # rest
    
    for qid in query_ids:
        if qid not in qrels:
            continue
        relevant = {doc_id for doc_id, score in qrels[qid].items() if score > 0}
        if not relevant:
            continue
        
        results = all_results.get(qid, [])
        r = recall_at_k(results, relevant, k=top_k)
        m = mrr(results, relevant)
        recalls.append(r)
        mrrs.append(m)
        
        # Query length stratification
        q_tokens = count_tokens(queries[qid])
        if q_tokens < 5:
            short_recalls.append(r)
        elif q_tokens <= 15:
            medium_recalls.append(r)
        else:
            long_recalls.append(r)
        
        # Gold passage length stratification
        has_long_gold = any(did in long_doc_ids for did in relevant)
        if has_long_gold:
            long_doc_recalls.append(r)
        else:
            other_doc_recalls.append(r)
    
    metrics = {
        "config": label,
        "overall": {
            "recall@10": round(np.mean(recalls), 4) if recalls else 0,
            "mrr": round(np.mean(mrrs), 4) if mrrs else 0,
            "num_queries": len(recalls)
        },
        "latency": {
            "cold_p50_ms": round(np.percentile(cold_latencies, 50), 2),
            "cold_p95_ms": round(np.percentile(cold_latencies, 95), 2),
            "warm_p50_ms": round(np.percentile(warm_latencies, 50), 2),
            "warm_p95_ms": round(np.percentile(warm_latencies, 95), 2),
        },
        "memory": {
            "peak_ram_mb": round(mem_after, 1),
            "delta_mb": round(mem_after - mem_before, 1)
        },
        "stratified": {
            "by_query_length": {
                "short_lt5": {
                    "recall@10": round(np.mean(short_recalls), 4) if short_recalls else 0,
                    "count": len(short_recalls)
                },
                "medium_5to15": {
                    "recall@10": round(np.mean(medium_recalls), 4) if medium_recalls else 0,
                    "count": len(medium_recalls)
                },
                "long_gt15": {
                    "recall@10": round(np.mean(long_recalls), 4) if long_recalls else 0,
                    "count": len(long_recalls)
                }
            },
            "by_gold_doc_length": {
                "top10pct_longest": {
                    "recall@10": round(np.mean(long_doc_recalls), 4) if long_doc_recalls else 0,
                    "count": len(long_doc_recalls)
                },
                "rest": {
                    "recall@10": round(np.mean(other_doc_recalls), 4) if other_doc_recalls else 0,
                    "count": len(other_doc_recalls)
                }
            }
        }
    }
    
    return metrics


def run_full_eval(methods: List[str] = None, rrf_k: int = 60, fetch_k: int = 100,
                  output_path: str = None):
    """Run evaluation across all methods and save results."""
    if methods is None:
        methods = ["bm25", "dense", "hybrid"]
    
    print("Loading FiQA dataset (dev split)...")
    corpus, queries, qrels = load_fiqa(split="dev")
    print(f"Corpus: {len(corpus)} docs | Queries: {len(queries)} | Qrels: {len(qrels)}")
    
    all_metrics = []
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Evaluating: {method}")
        print(f"{'='*60}")
        
        if method == "bm25":
            retriever = BM25Retriever()
            label = "BM25 (Okapi)"
        elif method == "dense":
            retriever = DenseRetriever()
            label = "Dense (all-MiniLM-L6-v2)"
        elif method == "hybrid":
            bm25 = BM25Retriever()
            dense = DenseRetriever()
            retriever = HybridRetriever(bm25, dense, rrf_k=rrf_k, fetch_k=fetch_k)
            label = f"Hybrid (RRF k={rrf_k}, fetch={fetch_k})"
        else:
            print(f"Unknown method: {method}, skipping")
            continue
        
        metrics = evaluate_retriever(
            retriever, queries, qrels, corpus,
            top_k=10, warmup_queries=100, label=label
        )
        all_metrics.append(metrics)
        
        print(f"\n  Recall@10: {metrics['overall']['recall@10']}")
        print(f"  MRR:       {metrics['overall']['mrr']}")
        print(f"  Warm p50:  {metrics['latency']['warm_p50_ms']:.1f} ms")
        print(f"  Warm p95:  {metrics['latency']['warm_p95_ms']:.1f} ms")
        print(f"  RAM:       {metrics['memory']['peak_ram_mb']:.0f} MB")
    
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "bench.json"
        )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    return all_metrics


def run_ablation(ablation_name: str, **kwargs):
    """Run a single ablation experiment."""
    print(f"Loading FiQA dataset (dev split)...")
    corpus, queries, qrels = load_fiqa(split="dev")
    
    if ablation_name == "rrf_k":
        # Ablation: vary RRF k parameter
        bm25 = BM25Retriever()
        dense = DenseRetriever()
        k_values = kwargs.get("k_values", [10, 30, 60, 100, 200])
        
        results = []
        for k in k_values:
            retriever = HybridRetriever(bm25, dense, rrf_k=k, fetch_k=100)
            metrics = evaluate_retriever(
                retriever, queries, qrels, corpus,
                top_k=10, warmup_queries=50,
                label=f"Hybrid RRF k={k}"
            )
            results.append(metrics)
            print(f"  RRF k={k}: Recall@10={metrics['overall']['recall@10']}, "
                  f"MRR={metrics['overall']['mrr']}")
        return results
    
    elif ablation_name == "fetch_k":
        # Ablation: vary fetch_k (number of candidates from each retriever)
        bm25 = BM25Retriever()
        dense = DenseRetriever()
        fetch_values = kwargs.get("fetch_values", [20, 50, 100, 200, 500])
        
        results = []
        for fk in fetch_values:
            retriever = HybridRetriever(bm25, dense, rrf_k=60, fetch_k=fk)
            metrics = evaluate_retriever(
                retriever, queries, qrels, corpus,
                top_k=10, warmup_queries=50,
                label=f"Hybrid fetch_k={fk}"
            )
            results.append(metrics)
            print(f"  fetch_k={fk}: Recall@10={metrics['overall']['recall@10']}, "
                  f"MRR={metrics['overall']['mrr']}")
        return results
    
    else:
        raise ValueError(f"Unknown ablation: {ablation_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval system")
    parser.add_argument("--methods", nargs="+", default=["bm25", "dense", "hybrid"],
                        help="Methods to evaluate")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF k parameter")
    parser.add_argument("--fetch-k", type=int, default=100, help="Fetch k for hybrid")
    parser.add_argument("--output", type=str, default=None, help="Output path for results")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["rrf_k", "fetch_k"],
                        help="Run ablation instead of full eval")
    
    args = parser.parse_args()
    
    if args.ablation:
        results = run_ablation(args.ablation)
        # Save ablation results
        out_path = args.output or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", f"ablation_{args.ablation}.json"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Ablation results saved to: {out_path}")
    else:
        run_full_eval(methods=args.methods, rrf_k=args.rrf_k,
                      fetch_k=args.fetch_k, output_path=args.output)
