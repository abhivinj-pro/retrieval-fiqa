"""Dense retriever using FAISS."""
from typing import List, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.index.dense_index import (
    DEFAULT_MODEL_KEY,
    artifact_path_for,
    load_dense_index,
    resolve_model,
)


class DenseRetriever:
    """Dense vector retrieval with sentence-transformers + FAISS."""

    def __init__(
        self,
        index: Optional[faiss.Index] = None,
        doc_ids: Optional[List[str]] = None,
        model: Optional[SentenceTransformer] = None,
        load_path: Optional[str] = None,
        model_key: Optional[str] = None,
        query_prefix: Optional[str] = None,
    ):
        """
        Args:
            index/doc_ids/model: in-memory components. If all three are
                supplied, loading is skipped.
            load_path: explicit artifact dir.
            model_key: short key selecting which pre-built artifact to
                load. Ignored if ``load_path`` is set.
            query_prefix: optional prefix prepended to every query before
                encoding. If ``None``, resolved from the model registry.
        """
        if index is not None and doc_ids is not None and model is not None:
            self.index = index
            self.doc_ids = doc_ids
            self.model = model
            self._resolved_model_name = None
        else:
            key = model_key or DEFAULT_MODEL_KEY
            path = load_path or artifact_path_for(key)
            self.index, self.doc_ids, self.model = load_dense_index(path)
            self._resolved_model_name = key

        if query_prefix is None:
            # Read instruction prefix from the registry using whichever
            # identifier we know about.
            ident = model_key or load_path or DEFAULT_MODEL_KEY
            _, _, query_prefix = resolve_model(ident)
        self.query_prefix = query_prefix or ""
    
    def _prep(self, queries: List[str]) -> List[str]:
        if not self.query_prefix:
            return queries
        return [self.query_prefix + q for q in queries]

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for relevant documents using dense retrieval."""
        query_embedding = self.model.encode(
            self._prep([query]), normalize_embeddings=True
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
            self._prep(queries), normalize_embeddings=True, batch_size=64
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
