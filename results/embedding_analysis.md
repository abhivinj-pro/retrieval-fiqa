# Findings: Dense Embedding Model Selection

**Decision:** `sentence-transformers/all-MiniLM-L6-v2` vs `BAAI/bge-small-en-v1.5`
---

## Decision

> **Adopt `all-MiniLM-L6-v2` (Dense) as the chosen operating point. Eliminate `bge-small-en-v1.5`.**
>
> BGE is marginally stronger on *unconstrained* quality (recall@10 +2.2% rel — inside the noise band; MRR +7.7% rel). But under this task's **binding constraints it is disqualified**: its measured warm **p95 latency is 122 ms against a ≤ 50 ms budget — 2.4× over**. MiniLM-Dense is the **only configuration in the entire benchmark that satisfies all three binding constraints**, and it delivers quality statistically indistinguishable from BGE on the ~80% of traffic that is medium-length.

---

## 1. Scope and method

- **Question:** Pick one dense encoder for the chosen operating point; eliminate the other.
- **Benchmark:** FiQA dev set, **500 queries**, Recall@10 / MRR / latency / peak RAM, with stratification by query length and gold-doc length.
- **Hardware:** 12-core CPU, 15.5 GB RAM, WSL2 (Linux), Python 3.11, **CPU-only** (matches serve constraint). Latency is hardware-dependent and reported as measured; quality metrics are hardware-independent.
- **Source data:** [results/bench.json](bench.json), [results/_bench_minilm.json](_bench_minilm.json), [results/_bench_bge.json](_bench_bge.json).

## 2. Binding constraints (from the task)

The chosen operating point **must satisfy all three**:

| Constraint | Budget |
|---|---|
| Compute | CPU only (no GPU at index or serve) |
| Latency | **warm p95 ≤ 50 ms / query** (after warmup) |
| Memory | **peak ≤ 2 GB RAM** at serve time |

> Configurations that miss a constraint are valid to *report*, but **cannot be the chosen operating point**.

## 3. Constraint-compliance matrix — all configurations

This is the table that decides the task. Latency = **warm p95**; memory = **peak RAM**.

| Configuration | Recall@10 | MRR | Warm p95 (ms) | Latency ≤ 50 ms | Peak RAM (MB) | RAM ≤ 2 GB | **Compliant?** |
|---|---:|---:|---:|:---:|---:|:---:|:---:|
| **Dense — MiniLM** | 0.4665 | 0.4532 | **43.22** | **PASS** | **1531.7** | **PASS** | **✅ YES** |
| Dense — BGE | 0.4767 | 0.4880 | 122.03 | FAIL | 1827.2 | PASS | **NO** |
| Hybrid — MiniLM | 0.4657 | 0.4616 | 531.85 | FAIL | 1944.8 | PASS | **NO** |
| Hybrid — BGE | 0.4723 | 0.4776 | 1505.33 | FAIL | 2183.9 | FAIL | **NO** |

**Headline result:** Of five configurations, **exactly one — MiniLM-Dense — clears all three binding constraints.** BGE-Dense is the second-best quality config but misses latency by 2.4×.

## 4. Dense head-to-head — quality

| Metric | MiniLM | BGE | Δ (abs) | Δ (rel) | Better |
|---|---:|---:|---:|---:|:---:|
| Recall@10 | 0.4665 | 0.4767 | +0.0102 | **+2.19%** | BGE |
| MRR | 0.4532 | 0.4880 | +0.0348 | **+7.68%** | BGE |

BGE wins both quality metrics. The **recall gap is small and within noise** (see §6); the **MRR gap is the more durable signal** — BGE ranks the correct passage higher, and that direction is consistent across both Dense and Hybrid runs.

## 5. Stratified quality — where the gap actually lives

**By query length (Dense):**

| Bucket | Count | MiniLM | BGE | Read |
|---|---:|---:|---:|---|
| short (<5 tok) | 13 | 0.4231 | 0.2692 | MiniLM "wins" — but **n=13, ±12 pp noise → ignore** |
| medium (5–15) | 403 | 0.4705 | 0.4704 | **Dead tie — ~80% of traffic** |
| long (>15) | 84 | 0.4543 | 0.5388 | **BGE +18.6% rel** — its real strength |

**By gold-doc length (Dense):**

| Bucket | Count | MiniLM | BGE | Read |
|---|---:|---:|---:|---|
| top 10% longest | 185 | 0.4140 | 0.4607 | BGE +11.3% rel |
| rest | 315 | 0.4974 | 0.4860 | MiniLM +2.3% rel |

**Interpretation:** The two encoders are **statistically tied on the medium-length queries that dominate the workload.** BGE's entire overall edge is concentrated in **long queries and long gold passages**, where richer semantics help. MiniLM's apparent short-query win rests on 13 queries and is noise. Since FiQA skews toward longer financial questions, BGE's strength is real — but it is a *narrow* slice of traffic, and it is bought at a latency cost the task does not permit.

## 6. Statistical honesty

- At n=500 with recall ≈ 0.47, the standard error is **±2.2 pp** (95% CI half-width ≈ ±4.4 pp). The Dense recall gap of **1.0 pp sits well inside one SE → not significant.** Recall@10 alone does **not** separate these models.
- The separation that *is* credible comes from (a) **MRR**, consistent in direction across two independent runs, and (b) the **long-query / long-doc strata** (≈1–1.5 SE — suggestive, not conclusive).
- The short-query stratum (n=13, SE ≈ ±12 pp) is statistically meaningless and must not influence the decision.

**Net:** Do not justify either model on overall Recall@10. The genuine, defensible quality difference is "BGE ranks slightly better, mostly on long inputs."

## 7. Latency and memory

| Metric | MiniLM | BGE | BGE penalty | Budget | Verdict |
|---|---:|---:|---:|---:|:---|
| Warm p50 (ms) | 27.08 | 56.95 | 2.10× | — | BGE p50 *alone* already > 50 ms |
| **Warm p95 (ms)** | **43.22** | **122.03** | **2.82×** | **≤ 50** | MiniLM 86% of budget; **BGE 244% — FAIL** |
| Cold p95 (ms) | 47.53 | 267.49 | 5.63× | — | BGE cold-start tail is severe |
| Peak RAM (MB) | 1531.7 | 1827.2 | +295.5 (+19%) | ≤ 2048 | Both pass; MiniLM has more headroom |

MiniLM clears the latency budget with ~14% headroom. BGE **fails on both warm p50 and warm p95** — the miss (2.4×) is far beyond any plausible hardware variance or the task's 5% reproducibility tolerance, so it is not a measurement artifact. On a CPU-only serve path, BGE's larger model and query-instruction prefix make sub-50 ms p95 unattainable without further optimization (quantization / ONNX / distillation) that is out of scope for this operating point.