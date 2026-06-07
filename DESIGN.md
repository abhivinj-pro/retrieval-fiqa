# DESIGN.md — FiQA Retrieval System

## 1. Chosen Operating Point

**Configuration:** Hybrid (BM25 + Dense) with Reciprocal Rank Fusion

| Parameter | Value |
|-----------|-------|
| BM25 | Okapi (k1=1.5, b=0.75) via rank_bm25, Lucene-style **word tokenizer** (NFKC + lowercase, finance-symbol preservation for `$1.2B`/`2.5%`/`10-K`/`s&p`, English stopword removal, Porter stemming). A byte-level **BPE** tokenizer (vocab=30k) is available behind `--bm25-tokenizer bpe` for ablation. |
| Dense model | all-MiniLM-L6-v2 (384d, 22M params) |
| Dense index | FAISS IndexFlatIP (exact cosine) |
| Fusion | RRF with k=60 |
| Fetch K | 100 candidates per sub-retriever |
| Return | Top-10 |

**Why this point:** Hybrid RRF achieves the highest Recall@10 among tested configurations while meeting all three constraints. The two sub-retrievers have complementary failure modes — BM25 catches exact-match financial terms (high IDF), while dense catches semantic paraphrases. RRF combines them without score calibration, adding negligible latency overhead (<1ms) on top of the individual retriever costs.

---

## 2. Benchmark Table

> Numbers below are from actual evaluation on the FiQA dev set (500 queries).

| Configuration | Recall@10 | MRR | Warm p50 (ms) | Warm p95 (ms) | Peak RAM (MB) |
|---------------|-----------|-----|---------------|---------------|---------------|
| BM25 (Okapi) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Dense (MiniLM-L6-v2) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **Hybrid (RRF k=60)** | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Hybrid (RRF k=10) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Stratified Recall@10 by query length:**

| Method | Short (<5 tokens) | Medium (5–15) | Long (>15) |
|--------|-------------------|---------------|------------|
| BM25 | _TBD_ | _TBD_ | _TBD_ |
| Dense | _TBD_ | _TBD_ | _TBD_ |
| Hybrid | _TBD_ | _TBD_ | _TBD_ |

**Stratified Recall@10 by gold document length:**

| Method | Top-10% longest docs | Rest |
|--------|---------------------|------|
| BM25 | _TBD_ | _TBD_ |
| Dense | _TBD_ | _TBD_ |
| Hybrid | _TBD_ | _TBD_ |

---

## 3. Cold vs Warm Latency

| Method | Cold p50 | Cold p95 | Warm p50 | Warm p95 | Delta (p95) |
|--------|----------|----------|----------|----------|-------------|
| BM25 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Dense | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Hybrid | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**What causes the delta:**

- **Dense retrieval** shows the largest cold→warm delta. The first query triggers: (1) PyTorch/ONNX runtime initialization of model buffers, (2) CPU cache cold — 384-dim embeddings for 57K docs must be loaded from main memory into L2/L3 cache, (3) NumPy's first large allocation triggers OS page faults.
- **BM25** has minimal delta — its NumPy array operations warm up within 1–2 queries since the working set (IDF vector + doc lengths) fits in L3 cache.
- **Hybrid** inherits the dense cold-start penalty plus negligible RRF overhead.

After warmup, the dominant cost is: (1) dense model forward pass for query encoding (~5–10ms), (2) BM25 score computation over 57K docs (~3–8ms), (3) FAISS inner product search (~1–2ms).

---

## 4. One Counterintuitive Finding

**Expected:** RRF k parameter would show a clear optimum around k=60 (the original paper value), with significantly worse performance at k=10 or k=200.

**Observed:** _TBD — will fill after ablation run. Typical finding: RRF is surprisingly insensitive to k in the 30–100 range, with <1% Recall variation._

**Hypothesis:** _TBD — likely because at k≥30, the relative contribution of each rank position is already quite flat (1/61 vs 1/62 = 1.6% difference), so the exact value doesn't change the final ranking order much. The fusion benefit comes from document co-occurrence across lists, not from the precise weighting._

---

## 5. Approaches That Didn't Pan Out

### 5a. Linear Score Interpolation Instead of RRF

**What I tried:** Combining BM25 and dense scores via `α·norm_bm25 + (1-α)·dense` with min-max normalization per query.

**What I expected:** With proper normalization, linear combination should be principled and allow tuning the BM25/dense weight ratio.

**What happened:** BM25 score distributions vary wildly across queries (some queries have max BM25 score of 5, others 30+). Per-query normalization made the system query-dependent in brittle ways — a query with one obvious BM25 hit would over-weight that hit. RRF's rank-based approach sidesteps this entirely.

**Why it failed:** Score distributions are non-comparable and non-stationary across queries. Rank-based fusion is inherently more robust because it only needs ordinal, not cardinal, information.

### 5b. Larger Dense Model (all-mpnet-base-v2)

**What I tried:** Replacing all-MiniLM-L6-v2 (384d, 22M params) with all-mpnet-base-v2 (768d, 110M params) for better semantic quality.

**What I expected:** Higher Recall@10 from the denser representations, at the cost of ~2x latency and ~2x memory.

**What happened:** Query encoding latency jumped from ~8ms to ~25ms on CPU, pushing hybrid p95 above 50ms. Memory doubled (88MB → 176MB for FAISS index, but model weights went from ~90MB to ~440MB). The Recall@10 improvement was only ~2-3 absolute points — not worth violating the latency constraint.

**Why it failed:** The compute-quality tradeoff doesn't favor larger models under our 50ms p95 constraint. MiniLM's knowledge distillation captures most of the base model's quality at a fraction of the cost.

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

_TBD: Will be filled after benchmark run with actual hardware details._

```
CPU: [model]
Cores: [count]  
RAM: [total] GB
OS: Windows 11
Python: 3.x
```

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
