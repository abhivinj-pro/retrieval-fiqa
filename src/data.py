"""Data loading utilities for BEIR FiQA dataset."""
import os
import json
from typing import Dict, Tuple
from beir import util
from beir.datasets.data_loader import GenericDataLoader


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")


def download_fiqa(data_dir: str = DATA_DIR) -> str:
    """Download FiQA dataset from BEIR. Returns path to dataset."""
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
    data_path = util.download_and_unzip(url, data_dir)
    return data_path


def load_fiqa(split: str = "dev", data_dir: str = DATA_DIR) -> Tuple[Dict, Dict, Dict]:
    """Load FiQA corpus, queries, and qrels for given split.
    
    Returns:
        corpus: {doc_id: {"title": str, "text": str}}
        queries: {query_id: str}
        qrels: {query_id: {doc_id: relevance_score}}
    """
    data_path = os.path.join(data_dir, "fiqa")
    if not os.path.exists(data_path):
        data_path = download_fiqa(data_dir)
    
    corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
    return corpus, queries, qrels


def get_corpus_texts(corpus: Dict) -> Tuple[list, list]:
    """Extract doc_ids and texts from corpus dict.
    
    Returns:
        doc_ids: list of document IDs
        texts: list of document text strings (title + text)
    """
    doc_ids = []
    texts = []
    for doc_id, doc in corpus.items():
        doc_ids.append(doc_id)
        title = doc.get("title", "").strip()
        text = doc.get("text", "").strip()
        combined = f"{title} {text}" if title else text
        texts.append(combined)
    return doc_ids, texts
