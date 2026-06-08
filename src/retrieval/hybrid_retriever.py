"""Hybrid retriever combining BM25 and Dense via Reciprocal Rank Fusion."""
from typing import Dict, List, Optional, Sequence, Tuple

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever


def reciprocal_rank_fusion(
    results_list: List[List[Tuple[str, float]]],
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[str, float]]:
    """Combine multiple ranked lists using (weighted) Reciprocal Rank Fusion.

    RRF score for document d:
        sum over lists i of  w_i * 1 / (k + rank_i(d))

    Unweighted RRF (``weights=None``) treats every retriever as equally
    trustworthy. On FiQA the dense retriever is much stronger than BM25
    (R@10 ~0.47 vs ~0.31), so symmetric fusion lets BM25's noisy tail
    demote dense's correct hits and the hybrid ends up *worse* than dense
    alone. Weighted RRF (e.g. ``weights=(0.3, 0.7)``) damps the weaker
    retriever's contribution and recovers (and exceeds) dense quality.

    Args:
        results_list: list of ranked result lists, each ``[(doc_id, score)]``.
        k: rank-discount constant (default 60, from the original paper).
            Smaller ``k`` makes top ranks count more.
        weights: per-list multiplicative weights. ``None`` -> all-ones.

    Returns:
        Merged ``[(doc_id, fused_score)]`` sorted by score descending.
    """
    if weights is None:
        weights = [1.0] * len(results_list)
    if len(weights) != len(results_list):
        raise ValueError(
            f"weights length {len(weights)} != results_list length {len(results_list)}"
        )

    fused_scores: Dict[str, float] = {}
    for w, results in zip(weights, results_list):
        if w == 0:
            continue
        for rank, (doc_id, _score) in enumerate(results, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + w / (k + rank)

    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


def _min_max_normalize(scores: Sequence[float]) -> List[float]:
    """Scale scores to [0, 1]. All-equal lists (e.g. all-zero BM25) map to 0."""
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi <= lo:
        return [0.0] * len(scores)
    span = hi - lo
    return [(s - lo) / span for s in scores]


def weighted_score_fusion(
    results_list: List[List[Tuple[str, float]]],
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[str, float]]:
    """Combine ranked lists by a weighted sum of per-list min-max-normalized scores.

    Unlike RRF, this preserves each retriever's *score magnitude* (after
    per-query normalization). RRF collapses every list to a fixed per-rank
    increment ``1/(k+rank)``, which throws away the very signal that makes a
    strong dense model useful: the large cosine gap between a confident top
    hit and the rest. When one retriever both dominates and emits calibrated
    scores (cosine similarity from a normalized dense model -- exactly the
    FiQA + dense regime here), keeping magnitudes lets dense stay on top while
    BM25 only boosts docs they agree on and adds lexical recall in the tail.

        final(d) = sum_i  w_i * minmax(scores_i)[d]   (absent from list_i -> 0)

    Args:
        results_list: list of ranked ``[(doc_id, score)]`` lists.
        weights: per-list weights. ``None`` -> all-ones.

    Returns:
        Merged ``[(doc_id, fused_score)]`` sorted by score descending.
    """
    if weights is None:
        weights = [1.0] * len(results_list)
    if len(weights) != len(results_list):
        raise ValueError(
            f"weights length {len(weights)} != results_list length {len(results_list)}"
        )

    fused_scores: Dict[str, float] = {}
    for w, results in zip(weights, results_list):
        if w == 0 or not results:
            continue
        normed = _min_max_normalize([s for _, s in results])
        for (doc_id, _score), ns in zip(results, normed):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + w * ns

    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """Hybrid retrieval combining BM25 + Dense with weighted RRF.

    Defaults are tuned for FiQA + MiniLM, where dense substantially
    outperforms BM25. The weighting and ``rrf_k`` push hybrid
    Recall@10 above the dense-only baseline. Both knobs are exposed so
    they can be re-tuned per dataset / dense model.
    """

    # Defaults chosen empirically (see results/bench.json ablations).
    DEFAULT_RRF_K = 60
    DEFAULT_WEIGHTS: Tuple[float, float] = (0.3, 0.7)  # (bm25, dense)
    DEFAULT_FETCH_K = 200
    DEFAULT_FUSION = "rrf"  # "rrf" | "score"; "score" preserves dense magnitudes

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        rrf_k: int = DEFAULT_RRF_K,
        fetch_k: int = DEFAULT_FETCH_K,
        weights: Sequence[float] = DEFAULT_WEIGHTS,
        fusion: str = DEFAULT_FUSION,
        drop_zero_bm25: bool = True,
    ):
        """
        Args:
            bm25_retriever: BM25Retriever instance
            dense_retriever: DenseRetriever instance
            rrf_k: RRF rank-discount (higher = flatter discount across ranks).
                Only used when ``fusion="rrf"``.
            fetch_k: how many results to fetch from each sub-retriever
                before fusion. A larger pool gives fusion more candidates to
                co-rank but increases BM25 scan time slightly.
            weights: ``(w_bm25, w_dense)`` weights applied inside fusion.
                Defaults bias toward the stronger dense retriever.
            fusion: ``"rrf"`` (reciprocal rank fusion -- rank only) or
                ``"score"`` (weighted sum of min-max-normalized scores --
                keeps dense's confidence gap). ``"score"`` is recommended
                whenever dense substantially outperforms BM25.
            drop_zero_bm25: discard BM25 hits with score <= 0 before fusion.
                BM25 pads its top-k with arbitrary zero-overlap docs once the
                real matches run out; fusing that random tail only displaces
                dense's borderline-correct hits.
        """
        if len(weights) != 2:
            raise ValueError(f"weights must be length 2, got {len(weights)}")
        if fusion not in ("rrf", "score"):
            raise ValueError(f"fusion must be 'rrf' or 'score', got {fusion!r}")
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k
        self.fetch_k = fetch_k
        self.weights = tuple(weights)
        self.fusion = fusion
        self.drop_zero_bm25 = drop_zero_bm25

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using hybrid BM25 + Dense fusion (RRF or normalized score)."""
        bm25_results = self.bm25.search(query, top_k=self.fetch_k)
        if self.drop_zero_bm25:
            bm25_results = [(d, s) for d, s in bm25_results if s > 0.0]
        dense_results = self.dense.search(query, top_k=self.fetch_k)

        if self.fusion == "score":
            fused = weighted_score_fusion(
                [bm25_results, dense_results],
                weights=self.weights,
            )
        else:
            fused = reciprocal_rank_fusion(
                [bm25_results, dense_results],
                k=self.rrf_k,
                weights=self.weights,
            )
        return fused[:top_k]

    def search_batch(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """Search multiple queries."""
        return [self.search(q, top_k) for q in queries]
