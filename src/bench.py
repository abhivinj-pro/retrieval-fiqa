"""Run all benchmarks: main eval + ablations."""
import json
import os
import sys
import platform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import run_full_eval, run_ablation


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


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 70)
    print("FIQA RETRIEVAL BENCHMARK")
    print("=" * 70)
    
    # System info
    sys_info = get_system_info()
    print(f"\nSystem: {sys_info['platform']}")
    print(f"CPU: {sys_info['processor']} ({sys_info['cpu_count']} cores)")
    print(f"RAM: {sys_info['ram_total_gb']} GB")
    
    # Main evaluation
    print("\n--- Main Evaluation ---")
    main_metrics = run_full_eval(
        methods=["bm25", "dense", "hybrid"],
        output_path=os.path.join(results_dir, "bench.json")
    )
    
    # Ablation: RRF k parameter
    print("\n\n--- Ablation: RRF k Parameter ---")
    rrf_ablation = run_ablation("rrf_k", k_values=[10, 30, 60, 100, 200])
    
    # Ablation: fetch_k parameter
    print("\n\n--- Ablation: Fetch K ---")
    fetch_ablation = run_ablation("fetch_k", fetch_values=[20, 50, 100, 200, 500])
    
    # Save combined results
    combined = {
        "system_info": sys_info,
        "main_eval": main_metrics,
        "ablations": {
            "rrf_k": rrf_ablation,
            "fetch_k": fetch_ablation
        }
    }
    
    with open(os.path.join(results_dir, "bench.json"), "w") as f:
        json.dump(combined, f, indent=2)
    
    print(f"\n\nAll results saved to: {results_dir}/bench.json")
    
    # Print summary table
    print("\n\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Config':<35} {'Recall@10':<12} {'MRR':<10} {'p95 (ms)':<10} {'RAM (MB)':<10}")
    print("-" * 70)
    for m in main_metrics:
        print(f"{m['config']:<35} {m['overall']['recall@10']:<12} "
              f"{m['overall']['mrr']:<10} {m['latency']['warm_p95_ms']:<10} "
              f"{m['memory']['peak_ram_mb']:<10}")


if __name__ == "__main__":
    main()
