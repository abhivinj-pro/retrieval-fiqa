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
from src.index.dense_index import MODEL_REGISTRY, DEFAULT_MODEL_KEY


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


def run_full_eval(methods: List[str] = None,
                  dense_model: str = DEFAULT_MODEL_KEY,
                  rrf_k: int = HybridRetriever.DEFAULT_RRF_K,
                  fetch_k: int = HybridRetriever.DEFAULT_FETCH_K,
                  weights: Tuple[float, float] = HybridRetriever.DEFAULT_WEIGHTS,
                  fusion: str = HybridRetriever.DEFAULT_FUSION,
                  output_path: str = None):
    """Run evaluation across the given methods and save results.

    Args:
        methods: subset of ``{"bm25", "dense", "hybrid"}``.
        dense_model: dense model key. Used for both the standalone
            ``"dense"`` method and the dense sub-retriever of ``"hybrid"``.
        rrf_k / fetch_k / weights: hybrid hyperparameters.
        output_path: where to write the JSON. If ``None``, defaults to
            ``results/bench.json``.
    """
    if methods is None:
        methods = ["bm25", "dense", "hybrid"]
    
    print("Loading FiQA dataset (dev split)...")
    corpus, queries, qrels = load_fiqa(split="dev")
    print(f"Corpus: {len(corpus)} docs | Queries: {len(queries)} | Qrels: {len(qrels)}")
    
    all_metrics = []
    hf_name = MODEL_REGISTRY[dense_model]["hf_name"]
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Evaluating: {method} (dense_model={dense_model})")
        print(f"{'='*60}")
        
        if method == "bm25":
            retriever = BM25Retriever()
            label = "BM25 (Okapi)"
        elif method == "dense":
            retriever = DenseRetriever(model_key=dense_model)
            label = f"Dense ({hf_name})"
        elif method == "hybrid":
            bm25 = BM25Retriever()
            dense = DenseRetriever(model_key=dense_model)
            retriever = HybridRetriever(bm25, dense, rrf_k=rrf_k,
                                        fetch_k=fetch_k, weights=weights,
                                        fusion=fusion)
            if fusion == "score":
                label = (f"Hybrid ({dense_model}, fusion=score, "
                         f"fetch={fetch_k}, w_bm25={weights[0]}, w_dense={weights[1]})")
            else:
                label = (f"Hybrid ({dense_model}, RRF k={rrf_k}, "
                         f"fetch={fetch_k}, w_bm25={weights[0]}, w_dense={weights[1]})")
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


def run_ablation(ablation_name: str, dense_model: str = DEFAULT_MODEL_KEY, **kwargs):
    """Run a single ablation experiment.

    Supported ablations:
        - ``"rrf_k"``: vary RRF k (defaults: [10, 30, 60, 100, 200])
        - ``"fetch_k"``: vary fetch_k (defaults: [20, 50, 100, 200, 500])
        - ``"weights"``: vary dense weight while BM25 weight=1
            (defaults: dense ∈ [0.5, 1.0, 1.5, 2.0, 3.0]); RRF fusion only.
        - ``"fusion"``: cross both fusion strategies (``"rrf"`` and ``"score"``)
            against the same dense-weight sweep as ``"weights"``. Produces a
            2 × N grid so per-weight rrf-vs-score comparisons are direct.
            Accepts ``dense_weights`` kwarg (default: [0.5, 1.0, 1.5, 2.0, 3.0]).
        - ``"dense_model"``: compare each model_key as standalone dense
    """
    print(f"Loading FiQA dataset (dev split)...")
    corpus, queries, qrels = load_fiqa(split="dev")

    if ablation_name == "rrf_k":
        bm25 = BM25Retriever()
        dense = DenseRetriever(model_key=dense_model)
        k_values = kwargs.get("k_values", [10, 30, 60, 100, 200])
        results = []
        for k in k_values:
            retriever = HybridRetriever(
                bm25, dense, rrf_k=k,
                fetch_k=HybridRetriever.DEFAULT_FETCH_K,
                weights=HybridRetriever.DEFAULT_WEIGHTS,
            )
            metrics = evaluate_retriever(
                retriever, queries, qrels, corpus,
                top_k=10, warmup_queries=50,
                label=f"Hybrid ({dense_model}) RRF k={k}"
            )
            results.append(metrics)
            print(f"  RRF k={k}: Recall@10={metrics['overall']['recall@10']}, "
                  f"MRR={metrics['overall']['mrr']}")
        return results

    elif ablation_name == "fetch_k":
        bm25 = BM25Retriever()
        dense = DenseRetriever(model_key=dense_model)
        fetch_values = kwargs.get("fetch_values", [20, 50, 100, 200, 500])
        results = []
        for fk in fetch_values:
            retriever = HybridRetriever(
                bm25, dense, rrf_k=HybridRetriever.DEFAULT_RRF_K,
                fetch_k=fk, weights=HybridRetriever.DEFAULT_WEIGHTS,
            )
            metrics = evaluate_retriever(
                retriever, queries, qrels, corpus,
                top_k=10, warmup_queries=50,
                label=f"Hybrid ({dense_model}) fetch_k={fk}"
            )
            results.append(metrics)
            print(f"  fetch_k={fk}: Recall@10={metrics['overall']['recall@10']}, "
                  f"MRR={metrics['overall']['mrr']}")
        return results

    elif ablation_name == "weights":
        bm25 = BM25Retriever()
        dense = DenseRetriever(model_key=dense_model)
        dense_weights = kwargs.get("dense_weights", [0.5, 1.0, 1.5, 2.0, 3.0])
        results = []
        for w in dense_weights:
            retriever = HybridRetriever(
                bm25, dense, rrf_k=HybridRetriever.DEFAULT_RRF_K,
                fetch_k=HybridRetriever.DEFAULT_FETCH_K,
                weights=(1.0, w),
            )
            metrics = evaluate_retriever(
                retriever, queries, qrels, corpus,
                top_k=10, warmup_queries=50,
                label=f"Hybrid ({dense_model}) w_bm25=1, w_dense={w}"
            )
            results.append(metrics)
            print(f"  w_dense={w}: Recall@10={metrics['overall']['recall@10']}, "
                  f"MRR={metrics['overall']['mrr']}")
        return results

    elif ablation_name == "fusion":
        bm25 = BM25Retriever()
        dense = DenseRetriever(model_key=dense_model)
        dense_weights = kwargs.get("dense_weights", [0.5, 1.0, 1.5, 2.0, 3.0])
        results = []
        for fusion_type in ("rrf", "score"):
            print(f"  -- fusion={fusion_type} --")
            for w in dense_weights:
                w_bm25, w_dense = 1.0, w
                retriever = HybridRetriever(
                    bm25, dense,
                    rrf_k=HybridRetriever.DEFAULT_RRF_K,
                    fetch_k=HybridRetriever.DEFAULT_FETCH_K,
                    weights=(w_bm25, w_dense),
                    fusion=fusion_type,
                )
                label = (
                    f"Hybrid ({dense_model}, fusion={fusion_type}, "
                    f"fetch={HybridRetriever.DEFAULT_FETCH_K}, "
                    f"w_bm25={w_bm25}, w_dense={w_dense})"
                )
                metrics = evaluate_retriever(
                    retriever, queries, qrels, corpus,
                    top_k=10, warmup_queries=50,
                    label=label,
                )
                results.append(metrics)
                print(f"    w_bm25={w_bm25}, w_dense={w_dense}: "
                      f"Recall@10={metrics['overall']['recall@10']}, "
                      f"MRR={metrics['overall']['mrr']}")
        return results

    elif ablation_name == "dense_model":
        keys = kwargs.get("model_keys", list(MODEL_REGISTRY.keys()))
        results = []
        for key in keys:
            hf_name = MODEL_REGISTRY[key]["hf_name"]
            retriever = DenseRetriever(model_key=key)
            metrics = evaluate_retriever(
                retriever, queries, qrels, corpus,
                top_k=10, warmup_queries=50,
                label=f"Dense ({hf_name})"
            )
            results.append(metrics)
            print(f"  {key}: Recall@10={metrics['overall']['recall@10']}, "
                  f"MRR={metrics['overall']['mrr']}")
        return results

    else:
        raise ValueError(f"Unknown ablation: {ablation_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval system")
    parser.add_argument("--methods", nargs="+", default=["bm25", "dense", "hybrid"],
                        help="Methods to evaluate")
    parser.add_argument("--dense-model", type=str, default=DEFAULT_MODEL_KEY,
                        choices=list(MODEL_REGISTRY.keys()),
                        help=f"Dense model key (default: {DEFAULT_MODEL_KEY})")
    parser.add_argument("--rrf-k", type=int, default=HybridRetriever.DEFAULT_RRF_K,
                        help="RRF k parameter")
    parser.add_argument("--fetch-k", type=int, default=HybridRetriever.DEFAULT_FETCH_K,
                        help="Fetch k for hybrid")
    parser.add_argument("--w-bm25", type=float,
                        default=HybridRetriever.DEFAULT_WEIGHTS[0],
                        help="Hybrid BM25 weight")
    parser.add_argument("--w-dense", type=float,
                        default=HybridRetriever.DEFAULT_WEIGHTS[1],
                        help="Hybrid dense weight")
    parser.add_argument("--fusion", type=str,
                        default=HybridRetriever.DEFAULT_FUSION,
                        choices=["rrf", "score"],
                        help=f"Hybrid fusion method (default: {HybridRetriever.DEFAULT_FUSION})")
    parser.add_argument("--output", type=str, default=None, help="Output path for results")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["rrf_k", "fetch_k", "weights", "dense_model"],
                        help="Run ablation instead of full eval")
    
    args = parser.parse_args()
    
    if args.ablation:
        results = run_ablation(args.ablation, dense_model=args.dense_model)
        out_path = args.output or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", f"ablation_{args.ablation}.json"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Ablation results saved to: {out_path}")
    else:
        run_full_eval(methods=args.methods,
                      dense_model=args.dense_model,
                      rrf_k=args.rrf_k,
                      fetch_k=args.fetch_k,
                      weights=(args.w_bm25, args.w_dense),
                      fusion=args.fusion,
                      output_path=args.output)
