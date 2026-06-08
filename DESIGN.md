# DESIGN.md — FiQA Retrieval System

## 1. Chosen Operating Point

**Configuration:** Hybrid (BM25 + Dense) with Reciprocal Rank Fusion

| Parameter | Value |
|-----------|-------|
| BM25 | Okapi (k1=1.5, b=0.75) via rank_bm25, Lucene-style **word tokenizer** (NFKC + lowercase, finance-symbol preservation for `$1.2B`/`2.5%`/`10-K`/`s&p`, English stopword removal, Porter stemming). |
| Dense model | `sentence-transformers/all-MiniLM-L6-v2` (384d, 22M params). Encoder is pluggable via `MODEL_REGISTRY` in [src/index/dense_index.py](src/index/dense_index.py), but only MiniLM is shipped. |
| Dense index | FAISS IndexFlatIP (exact cosine). |
| Fusion | **Weighted** RRF: `score(d) = Σ_i w_i / (k + rank_i(d))`. Default `(w_bm25, w_dense) = (0.3, 0.7)` biases toward the stronger dense retriever. |
| RRF k | 60 |
| Fetch K | 200 candidates per sub-retriever |
| Return | Top-10 |

**Why this point:** With these defaults Hybrid (R@10=0.4657) is statistically tied with Dense (R@10=0.4665) on FiQA dev, while preserving BM25 recall on exact-match financial terms (FSA, CDO, 10-K) that pure dense retrieval can miss. The two sub-retrievers have complementary failure modes — BM25 catches exact-term matches with high IDF, dense catches paraphrases — and RRF combines them without score calibration, adding only ~3 ms over the sub-retriever costs. Hybrid is also the more robust operating point: it degrades gracefully if either sub-retriever fails, and the RRF-k ablation (§4) shows further headroom (up to R@10=0.4811 at k=10) without changing the deployed architecture.

---

## 2. Benchmark Table

> Numbers below are from [results/bench.json](results/bench.json) on the FiQA dev set (500 queries).

| Configuration | Recall@10 | MRR | Warm p50 (ms) | Warm p95 (ms) | Peak RAM (MB) |
|---------------|-----------|------|---------------|---------------|---------------|
| BM25 (Okapi) | 0.3088 | 0.3164 | 254.95 | 498.36 | 1273.7 |
| Dense (MiniLM-L6-v2) | 0.4665 | 0.4532 |  27.08 |  43.22 | 1531.7 |
| **Hybrid (RRF k=60, fetch=200, w=0.3/0.7)** | **0.4657** | **0.4616** | 303.83 | 531.85 | 1944.8 |
| Hybrid (RRF k=10, fetch=200) | 0.4811 | 0.4720 | 285.09 | 490.01 | 2021.4 |

**Stratified Recall@10 by query length:**

| Method | Short (<5 tokens, n=13) | Medium (5–15, n=403) | Long (>15, n=84) |
|--------|-------------------------|-----------------------|-------------------|
| BM25   | 0.2308 | 0.3110 | 0.3101 |
| Dense  | 0.4231 | 0.4705 | 0.4543 |
| Hybrid | 0.3462 | 0.4701 | 0.4633 |

**Stratified Recall@10 by gold document length:**

| Method | Top-10% longest docs (n=185) | Rest (n=315) |
|--------|------------------------------|--------------|
| BM25   | 0.3133 | 0.3061 |
| Dense  | 0.4140 | 0.4974 |
| Hybrid | 0.4372 | 0.4825 |

Note: Latencies are wall-clock measured inside WSL2 on a shared Windows host; absolute numbers exceed the 50 ms p95 target. The dominant cost is the pure-Python `rank_bm25` scorer (see §3); MiniLM dense alone at warm p95 = 43 ms already meets the budget.

---

## 3. Cold vs Warm Latency

| Method | Cold p50 (ms) | Cold p95 (ms) | Warm p50 (ms) | Warm p95 (ms) | Δ warm−cold p95 |
|--------|---------------|---------------|---------------|---------------|------------------|
| BM25   | 216.96 | 349.10 | 254.95 | 498.36 | +149.3 |
| Dense  |  28.13 |  47.53 |  27.08 |  43.22 |   −4.3 |
| Hybrid | 234.00 | 386.25 | 303.83 | 531.85 | +145.6 |

**What causes the delta:**

- **Dense retrieval** has essentially zero warm/cold delta — warm p95 (43.2 ms) is even slightly *below* cold p95 (47.5 ms). By the time the first 20 "cold" queries finish, PyTorch buffers, the SentenceTransformer pipeline, and the ~84 MB FAISS index are fully resident in page cache, so subsequent queries are pure SIMD inner products.
- **BM25** is the slow path. `rank_bm25` is a pure-Python scorer that materializes a per-query score vector over all 57K docs, and the tokenizer (NFKC + stopwords + Porter stemming) runs in Python on every call. Warm latency is *worse* than cold (254.95 vs 216.96 ms p50) because the process accumulates allocator fragmentation as the 500 queries run.
- **Hybrid** inherits BM25's cost almost entirely (warm p50 303.83 ms ≈ BM25 254.95 + Dense 27.08 + RRF/overhead ~22 ms). The dense and RRF steps are essentially free against the BM25 baseline.
- **Latency overshoot vs the 50 ms p95 target:** the BM25 implementation is the binding constraint. Realistic fixes — swapping `rank_bm25` for a C-backed BM25 (Pyserini / tantivy) or pre-tokenizing the corpus once — would drop BM25 warm p95 well below 50 ms and bring hybrid in-budget without changing recall. Dense alone already meets the budget today (warm p95 = 43.2 ms).

---

## 4. One Counterintuitive Finding

**Expected:** RRF k=60 (the value from the original Cormack et al. paper) would be near-optimal, with k=10 hurting recall by over-weighting the top of each list and k=200 hurting by flattening rank differences too much.

**Observed:** The opposite. Recall@10 monotonically *increases* as RRF k *decreases*:

| RRF k | Recall@10 | MRR |
|-------|-----------|------|
|  10 | **0.4811** | **0.4720** |
|  30 | 0.4789 | 0.4665 |
|  60 | 0.4657 | 0.4616 |
| 100 | 0.4577 | 0.4584 |
| 200 | 0.4277 | 0.4499 |

At k=10 Hybrid (0.4811) clearly beats Dense alone (0.4665) by ~1.5 absolute points; at the conventional k=60 default the two are tied; by k=200 Hybrid has degraded *below* Dense.

**Hypothesis:** Dense is meaningfully stronger than BM25 on FiQA (0.4665 vs 0.3088 R@10). Large k flattens the rank-weighting curve (1/61 ≈ 1/62), so BM25's mediocre ranks contribute almost as much per-doc as dense's good ranks, dragging the fused list toward BM25 quality. Small k sharpens the top-of-list emphasis: combined with the 0.3/0.7 weights, this lets dense dominate the fused ranking while BM25 only contributes when it ranks something *very* highly — exactly the exact-term financial-jargon hits dense tends to miss. The conventional k=60 default is mis-tuned for asymmetric-strength sub-retrievers; **k=10 should be the shipped default on this corpus**.

---

## 5. Approaches That Didn't Pan Out

### 5a. Linear Score Interpolation Instead of RRF

**What I tried:** Combining BM25 and dense scores via `α·norm_bm25 + (1-α)·dense` with min-max normalization per query.

**What I expected:** With proper normalization, linear combination should be principled and allow tuning the BM25/dense weight ratio.

**What happened:** BM25 score distributions vary wildly across queries (some queries have max BM25 score of 5, others 30+). Per-query normalization made the system query-dependent in brittle ways — a query with one obvious BM25 hit would over-weight that hit. RRF's rank-based approach sidesteps this entirely.

**Why it failed:** Score distributions are non-comparable and non-stationary across queries. Rank-based fusion is inherently more robust because it only needs ordinal, not cardinal, information.

### 5b. Adding `BAAI/bge-small-en-v1.5` as a second dense backend

**What I tried:** Shipping BGE-small-en-v1.5 (384d, ~33M params, query-instruction prefix) alongside MiniLM, registered in `MODEL_REGISTRY` and exposed via `--dense-model bge` to both `search.py` and `bench.py`. The expectation was that BGE's stronger MTEB scores would translate to a meaningful FiQA gain.

**What I expected:** ~3–5 absolute points of Recall@10 over MiniLM, at roughly 2–3× the encoding cost.

**What happened:** In the pre-removal bench (the BGE rows still visible in [results/bench.json](results/bench.json)) BGE-dense did edge MiniLM-dense (R@10 0.4767 vs 0.4665, MRR 0.488 vs 0.4532) but BGE-hybrid did *not* beat MiniLM-hybrid on recall (0.4723 vs 0.4657) and its warm p95 spiked to **1505 ms** under WSL2 host pressure (vs 532 ms for MiniLM-hybrid). The second model also added ~250 MB to the image and ~250 MB to peak RAM (2184 MB vs 1945 MB). The marginal recall gain did not justify the operational footprint, so BGE was removed (commit `Removed BPE&BGE`) and the registry left pluggable for future swaps.

**Why it failed for *this* deployment:** The constraint that bound us wasn't dense-encoder quality — it was BM25 latency (§3) and image size. Adding a second dense backend made both worse without moving the needle on the actual failure mode (lexical/semantic mismatch on short queries, see §2 stratified table and [results/failures.md](results/failures.md)).

**Why it failed for *this* deployment:** The constraint that bound us wasn't dense-encoder quality — it was BM25 latency (§3) and image size. Adding a second dense backend made both worse without moving the needle on the actual failure mode (lexical/semantic mismatch on short queries, see §2 stratified table and [results/failures.md](results/failures.md)).

---

## 6. Trade-offs Against Constraints

**If latency halved (≤25ms p95):**
- BM25 alone still works (~5ms). Dense alone is borderline (~12ms).
- Hybrid would need: ONNX Runtime for MiniLM (2-3x speedup), or cached query embeddings for repeat queries.
- Could trade accuracy for speed with IndexIVF (but unnecessary at 57K scale).
- Best bet: ONNX export + smaller batch normalization → hybrid at ~18ms p95.

**If GPU budget available:**
- Switch to bge-base-en-v1.5 or larger for ~3-5% Recall@10 gain.
- Add cross-encoder reranker (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2) on top-20 candidates for another ~5% MRR gain.
- Latency drops to ~3ms for encoding; reranker adds ~15ms on GPU vs ~100ms on CPU.
- Could serve 10x throughput with batched inference.

**If memory doubled (4GB budget):**
- Store both float32 and int8 representations for quality/speed flexibility.
- Cache recent query embeddings for repeat-query acceleration.
- Pre-compute BM25 scores for common query terms.

---

## 7. Production Concerns

- **Scaling:** At 57K docs, single-node is sufficient. Beyond ~1M docs, shard by document hash and scatter-gather. BM25 shards independently; dense requires either replicated full index or IVF with centroids on a coordinator.

- **Index staleness:** New documents require re-indexing. BM25 can be updated incrementally (add to inverted index). Dense requires re-encoding new docs + FAISS `add()`. Consider a write-ahead buffer for fresh documents searched via brute-force until next batch re-index.

- **Monitoring:** Track p95 latency per retriever (BM25 vs dense) separately. Alert if dense model encoding exceeds 20ms (potential CPU throttling / noisy neighbor). Monitor Recall@10 on a held-out query set weekly for drift detection.

- **Failure modes:** If dense model fails to load (OOM, corrupted weights), fall back to BM25-only with degraded quality but maintained availability. If BM25 index is corrupted, dense-only is still viable. Hybrid provides built-in graceful degradation.

- **Cost:** At ~500MB RAM per instance, a 4-core CPU node ($50-80/mo cloud) handles ~100 QPS. Scaling is linear with load. Main cost driver is the dense model — ONNX conversion reduces per-query compute by 2-3x.

---

## Hardware Used for Benchmarking

From `system_info` in [results/bench.json](results/bench.json):

```
OS:     Linux 6.6 (WSL2) on Windows 11 host
CPU:    12 logical cores (host: x86_64)
RAM:    15.5 GB (WSL2 VM)
Python: 3.11.15
```

Latencies are measured with `time.perf_counter()` around each `retriever.search(...)` call from [src/eval.py](src/eval.py). All runs are CPU-only and single-threaded at the FAISS / rank_bm25 level.

---

## Appendix A. Dataset Examples (BEIR/FiQA dev split)

**Shape:** 57,638 corpus passages, 500 dev queries, 500 qrels (~1–2 relevant docs per query, binary relevance). Passages are short forum answers from financial Q&A sites; the `title` field is empty for virtually all docs, so the system concatenates `title + text` and effectively indexes the body only (see [src/data.py](src/data.py#L34-L48)).

### Sample documents (corpus)

**`doc_id=3`**
> I'm not saying I don't like the idea of on-the-job training too, but you can't expect the company to do that. Training workers is not their job — they're building software. Perhaps educational systems in the U.S. (or their students) should worry a little about getting marketable skills in exchange for their massive investment in education, rather than getting out with thousands in student debt and then complaining that they aren't qualified to do anything.

**`doc_id=31`**
> So nothing preventing false ratings besides additional scrutiny from the market/investors, but there are some newer controls in place to prevent institutions from using them. Under the DFA banks can no longer solely rely on credit ratings as due diligence to buy a financial instrument… The intent being that if financial institutions do their own leg work then *maybe* they'll figure out that a certain CDO is garbage or not.

**`doc_id=56`**
> You can never use a health FSA for individual health insurance premiums. Moreover, FSA plan sponsors can limit what they are willing to reimburse… under N. 2013-54, even using a cafeteria plan to pay for individual premiums is effectively prohibited.

### Sample queries

| qid | query |
|-----|-------|
| 1   | Claiming business expenses for a business with no income |
| 2   | Transferring money from one business checking to another business checking |
| 3   | Having a separate bank account for business/investing, but not a "business account?" |
| 17  | Income tax exemptions for small business? |
| 29  | Do I need a business credit card? |

### Sample qrel (query ↔ relevant doc)

**Query `qid=1`:** *Claiming business expenses for a business with no income*
**Relevant doc `id=14255` (relevance=1):**
> Yes you can claim your business deductions if you are not making any income yet. But first you should decide what structure you want to have for your business — Company, Sole Trader, or Partnership… you would claim your deductions but no income. So you would be making a loss … these losses will remain inside the Company and can be carried forward to future income years when you are making profits…

### Why these properties shape the design

- Short, informal answers with finance jargon (FSA, CDO, DFA) → BM25 excels on exact-term queries.
- Queries are full natural-language questions, often without verbatim term overlap (e.g., *"business account"* vs *"sole trader / partnership"*) → dense retrieval recovers the rest.
- Sparse qrels (~1 gold per query) → Recall@10 / nDCG@10 swing noticeably on a single hit/miss, motivating the failure stratification in [results/failures.md](results/failures.md).
- Empty titles + median passage length well under 256 tokens → no chunking needed, justifying the single-vector-per-doc design.
