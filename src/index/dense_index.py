"""Dense indexing using sentence-transformers + FAISS.

One dense model is supported:

* ``"minilm"`` -> ``sentence-transformers/all-MiniLM-L6-v2`` (384d, ~80MB).
  Persisted to ``index_artifacts/dense/``.
"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "index_artifacts")


# Registry: model_key -> (hf_model_name, artifact_subdir, query_prefix)
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "minilm": {
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "subdir": "dense",
        "query_prefix": "",
    },
}

DEFAULT_MODEL_KEY = "minilm"
DEFAULT_MODEL = MODEL_REGISTRY[DEFAULT_MODEL_KEY]["hf_name"]


def resolve_model(model_key_or_name: str) -> Tuple[str, str, str]:
    """Map a short key OR an HF model name to (hf_name, subdir, query_prefix).

    Unknown HF names fall back to a slugified subdir so arbitrary models
    are still buildable, just without a query prefix.
    """
    if model_key_or_name in MODEL_REGISTRY:
        cfg = MODEL_REGISTRY[model_key_or_name]
        return cfg["hf_name"], cfg["subdir"], cfg["query_prefix"]
    for cfg in MODEL_REGISTRY.values():
        if cfg["hf_name"] == model_key_or_name:
            return cfg["hf_name"], cfg["subdir"], cfg["query_prefix"]
    slug = "dense_" + model_key_or_name.split("/")[-1].lower().replace("-", "_")
    return model_key_or_name, slug, ""


def artifact_path_for(model_key_or_name: str) -> str:
    _, subdir, _ = resolve_model(model_key_or_name)
    return os.path.join(INDEX_DIR, subdir)


def build_dense_index(
    doc_ids: List[str],
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 256,
    save_path: Optional[str] = None,
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
        save_path = artifact_path_for(model_name)
    os.makedirs(save_path, exist_ok=True)
    
    faiss.write_index(index, os.path.join(save_path, "faiss.index"))
    np.save(os.path.join(save_path, "doc_ids.npy"), np.array(doc_ids))
    
    # Save model name for loading
    with open(os.path.join(save_path, "config.txt"), "w") as f:
        f.write(model_name)
    
    return index, doc_ids, model


def load_dense_index(load_path: Optional[str] = None,
                     model_key: Optional[str] = None
                     ) -> Tuple[faiss.Index, List[str], SentenceTransformer]:
    """Load pre-built dense index and model.

    Args:
        load_path: explicit artifact dir. If ``None``, derive from
            ``model_key`` (or fall back to the legacy ``index_artifacts/dense``).
        model_key: short key selecting which pre-built artifact to load.
            Ignored if ``load_path`` is set.
    """
    if load_path is None:
        key = model_key or DEFAULT_MODEL_KEY
        load_path = artifact_path_for(key)
    
    index = faiss.read_index(os.path.join(load_path, "faiss.index"))
    doc_ids = np.load(os.path.join(load_path, "doc_ids.npy"), allow_pickle=True).tolist()
    
    with open(os.path.join(load_path, "config.txt"), "r") as f:
        model_name = f.read().strip()
    
    model = SentenceTransformer(model_name)
    return index, doc_ids, model
