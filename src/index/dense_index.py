"""Dense indexing using sentence-transformers + FAISS."""
import os
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "index_artifacts")

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def build_dense_index(
    doc_ids: List[str],
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 256,
    save_path: str = None
) -> Tuple[faiss.Index, List[str], SentenceTransformer]:
    """Build FAISS flat index from document embeddings.
    
    Args:
        doc_ids: document identifiers
        texts: document texts to encode
        model_name: sentence-transformers model name
        batch_size: encoding batch size
        save_path: path to save index artifacts
    
    Returns:
        index: FAISS index
        doc_ids: ordered document IDs
        model: loaded SentenceTransformer model
    """
    model = SentenceTransformer(model_name)
    
    print(f"Encoding {len(texts)} documents with {model_name}...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    embeddings = embeddings.astype(np.float32)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim since normalized)
    index.add(embeddings)
    
    if save_path is None:
        save_path = os.path.join(INDEX_DIR, "dense")
    os.makedirs(save_path, exist_ok=True)
    
    faiss.write_index(index, os.path.join(save_path, "faiss.index"))
    np.save(os.path.join(save_path, "doc_ids.npy"), np.array(doc_ids))
    
    # Save model name for loading
    with open(os.path.join(save_path, "config.txt"), "w") as f:
        f.write(model_name)
    
    return index, doc_ids, model


def load_dense_index(load_path: str = None) -> Tuple[faiss.Index, List[str], SentenceTransformer]:
    """Load pre-built dense index and model."""
    if load_path is None:
        load_path = os.path.join(INDEX_DIR, "dense")
    
    index = faiss.read_index(os.path.join(load_path, "faiss.index"))
    doc_ids = np.load(os.path.join(load_path, "doc_ids.npy"), allow_pickle=True).tolist()
    
    with open(os.path.join(load_path, "config.txt"), "r") as f:
        model_name = f.read().strip()
    
    model = SentenceTransformer(model_name)
    return index, doc_ids, model
