"""BM25 retriever."""
from typing import List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from src.index.bm25_index import TokenizerLike, tokenize, load_bm25_index


class BM25Retriever:
    """BM25 retrieval over pre-built index."""

    def __init__(
        self,
        bm25: Optional[BM25Okapi] = None,
        doc_ids: Optional[List[str]] = None,
        tokenizer: Optional[TokenizerLike] = None,
        load_path: Optional[str] = None,
    ):
        if bm25 is not None and doc_ids is not None and tokenizer is not None:
            self.bm25 = bm25
            self.doc_ids = doc_ids
            self.tokenizer = tokenizer
        else:
            self.bm25, self.doc_ids, self.tokenizer = load_bm25_index(load_path)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for relevant documents.
        
        Args:
            query: search query string
            top_k: number of results to return
        
        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        query_tokens = tokenize(query, self.tokenizer)
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(self.doc_ids[i], float(scores[i])) for i in top_indices]
        return results
    
    def search_batch(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """Search multiple queries."""
        return [self.search(q, top_k) for q in queries]
