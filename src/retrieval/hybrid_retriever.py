"""Hybrid retriever combining BM25 and Dense via Reciprocal Rank Fusion."""
from typing import List, Tuple, Dict

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever


def reciprocal_rank_fusion(
    results_list: List[List[Tuple[str, float]]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.
    
    RRF score for document d = sum over lists of 1 / (k + rank(d))
    
    Args:
        results_list: list of ranked result lists, each containing (doc_id, score)
        k: RRF parameter controlling rank discount (default 60, from original paper)
    
    Returns:
        Merged results sorted by RRF score descending
    """
    fused_scores: Dict[str, float] = {}
    
    for results in results_list:
        for rank, (doc_id, _score) in enumerate(results, start=1):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
            fused_scores[doc_id] += 1.0 / (k + rank)
    
    sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


class HybridRetriever:
    """Hybrid retrieval combining BM25 + Dense with RRF."""
    
    def __init__(self, bm25_retriever: BM25Retriever, dense_retriever: DenseRetriever,
                 rrf_k: int = 60, fetch_k: int = 100):
        """
        Args:
            bm25_retriever: BM25Retriever instance
            dense_retriever: DenseRetriever instance  
            rrf_k: RRF parameter (higher = less weight to top ranks)
            fetch_k: how many results to fetch from each sub-retriever before fusion
        """
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k
        self.fetch_k = fetch_k
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using hybrid BM25 + Dense with RRF fusion.
        
        Args:
            query: search query string
            top_k: number of results to return
        
        Returns:
            List of (doc_id, rrf_score) tuples
        """
        bm25_results = self.bm25.search(query, top_k=self.fetch_k)
        dense_results = self.dense.search(query, top_k=self.fetch_k)
        
        fused = reciprocal_rank_fusion([bm25_results, dense_results], k=self.rrf_k)
        return fused[:top_k]
    
    def search_batch(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """Search multiple queries."""
        return [self.search(q, top_k) for q in queries]
