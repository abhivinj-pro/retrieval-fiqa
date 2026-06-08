"""Run all benchmarks: main eval (BM25 + each dense model + each hybrid) + ablations."""
import json
import os
import sys
import platform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import run_full_eval, run_ablation
from src.index.dense_index import MODEL_REGISTRY, artifact_path_for
from src.retrieval.hybrid_retriever import HybridRetriever


def get_system_info():
    """Collect system information for reproducibility."""
    import psutil
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "python_version": platform.python_version()
    }


def _available_models():
    """Only run dense/hybrid for models whose index actually exists."""
    return [k for k in MODEL_REGISTRY
            if os.path.exists(os.path.join(artifact_path_for(k), "faiss.index"))]


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 70)
    print("FIQA RETRIEVAL BENCHMARK")
    print("=" * 70)
    
    sys_info = get_system_info()
    print(f"\nSystem: {sys_info['platform']}")
    print(f"CPU: {sys_info['processor']} ({sys_info['cpu_count']} cores)")
    print(f"RAM: {sys_info['ram_total_gb']} GB")

    available = _available_models()
    print(f"\nDense models with indexes present: {available}")
    if not available:
        raise RuntimeError(
            "No dense indexes found. Run `python src/build_index.py` first."
        )

    # ----- Main eval -----
    # BM25 is shared, evaluated once. Then per-model dense + hybrid.
    print("\n--- Main Evaluation: BM25 ---")
    main_metrics = run_full_eval(
        methods=["bm25"], dense_model=available[0],
        output_path=os.path.join(results_dir, "bench.json"),
    )
    for key in available:
        print(f"\n--- Main Evaluation: dense+hybrid for '{key}' ---")
        m = run_full_eval(
            methods=["dense", "hybrid"], dense_model=key,
            output_path=os.path.join(results_dir, f"_bench_{key}.json"),
        )
        main_metrics.extend(m)

    # ----- Ablations (run on the default dense model only) -----
    default_key = available[0]
    print(f"\n\n--- Ablation: RRF k Parameter (dense={default_key}) ---")
    rrf_ablation = run_ablation("rrf_k", dense_model=default_key,
                                k_values=[10, 30, 60, 100, 200])

    print(f"\n\n--- Ablation: Fetch K (dense={default_key}) ---")
    fetch_ablation = run_ablation("fetch_k", dense_model=default_key,
                                  fetch_values=[20, 50, 100, 200, 500])

    print(f"\n\n--- Ablation: Hybrid Dense Weight (dense={default_key}) ---")
    weights_ablation = run_ablation("weights", dense_model=default_key,
                                    dense_weights=[0.5, 1.0, 1.5, 2.0, 3.0])

    print(f"\n\n--- Ablation: Fusion Strategy (dense={default_key}) ---")
    fusion_ablation = run_ablation("fusion", dense_model=default_key)

    ablations = {
        "rrf_k": rrf_ablation,
        "fetch_k": fetch_ablation,
        "weights": weights_ablation,
        "fusion": fusion_ablation,
    }

    if len(available) > 1:
        print(f"\n\n--- Ablation: Dense Model Comparison ---")
        ablations["dense_model"] = run_ablation("dense_model", model_keys=available)

    # ----- Combined output -----
    combined = {
        "system_info": sys_info,
        "main_eval": main_metrics,
        "ablations": ablations,
    }
    
    with open(os.path.join(results_dir, "bench.json"), "w") as f:
        json.dump(combined, f, indent=2)

    # Clean up the per-model scratch files now that they're folded into bench.json.
    for key in available:
        scratch = os.path.join(results_dir, f"_bench_{key}.json")
        if os.path.exists(scratch):
            os.remove(scratch)
    
    print(f"\n\nAll results saved to: {results_dir}/bench.json")
    
    print("\n\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Config':<55} {'Recall@10':<12} {'MRR':<10} {'p95 (ms)':<10} {'RAM (MB)':<10}")
    print("-" * 100)
    for m in main_metrics:
        print(f"{m['config']:<55} {m['overall']['recall@10']:<12} "
              f"{m['overall']['mrr']:<10} {m['latency']['warm_p95_ms']:<10} "
              f"{m['memory']['peak_ram_mb']:<10}")


if __name__ == "__main__":
    main()
