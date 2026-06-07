"""Dense retriever using FAISS."""
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.index.dense_index import load_dense_index


class DenseRetriever:
    """Dense vector retrieval with sentence-transformers + FAISS."""
    
    def __init__(self, index: faiss.Index = None, doc_ids: List[str] = None,
                 model: SentenceTransformer = None, load_path: str = None):
        if index is not None and doc_ids is not None and model is not None:
            self.index = index
            self.doc_ids = doc_ids
            self.model = model
        else:
            self.index, self.doc_ids, self.model = load_dense_index(load_path)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for relevant documents using dense retrieval.
        
        Args:
            query: search query string
            top_k: number of results to return
        
        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for missing results
                results.append((self.doc_ids[idx], float(score)))
        return results
    
    def search_batch(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """Search multiple queries (batched encoding)."""
        query_embeddings = self.model.encode(
            queries, normalize_embeddings=True, batch_size=64
        ).astype(np.float32)
        
        scores, indices = self.index.search(query_embeddings, top_k)
        
        all_results = []
        for q_scores, q_indices in zip(scores, indices):
            results = []
            for score, idx in zip(q_scores, q_indices):
                if idx >= 0:
                    results.append((self.doc_ids[idx], float(score)))
            all_results.append(results)
        return all_results
