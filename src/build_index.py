"""Build all indexes for the retrieval system."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_fiqa, get_corpus_texts
from src.index.bm25_index import build_bm25_index, DEFAULT_TOKENIZER_KIND
from src.index.dense_index import build_dense_index, DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(description="Build retrieval indexes")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Dense model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Encoding batch size (default: 256)")
    parser.add_argument("--bm25-tokenizer", type=str, default=DEFAULT_TOKENIZER_KIND,
                        choices=["word", "bpe"],
                        help=(
                            "BM25 tokenizer: 'word' (default) uses the "
                            "Lucene-style English pipeline with finance-symbol "
                            "preservation, stopwords, and Porter stemming; "
                            "'bpe' uses a byte-level BPE tokenizer trained on "
                            "the corpus (kept for ablation)."
                        ))
    parser.add_argument("--skip-bm25", action="store_true", help="Skip BM25 index build")
    parser.add_argument("--skip-dense", action="store_true", help="Skip dense index build")
    
    args = parser.parse_args()
    
    print("Loading FiQA corpus...")
    corpus, _, _ = load_fiqa()
    doc_ids, texts = get_corpus_texts(corpus)
    print(f"Loaded {len(doc_ids)} documents")
    
    if not args.skip_bm25:
        print(f"\n--- Building BM25 Index (tokenizer={args.bm25_tokenizer}) ---")
        start = time.time()
        build_bm25_index(doc_ids, texts, tokenizer_kind=args.bm25_tokenizer)
        elapsed = time.time() - start
        print(f"BM25 index built in {elapsed:.1f}s")
    
    if not args.skip_dense:
        print(f"\n--- Building Dense Index ({args.model}) ---")
        start = time.time()
        build_dense_index(doc_ids, texts, model_name=args.model, batch_size=args.batch_size)
        elapsed = time.time() - start
        print(f"Dense index built in {elapsed:.1f}s")
    
    print("\nAll indexes built successfully!")
    print(f"Artifacts saved to: {os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'index_artifacts')}")


if __name__ == "__main__":
    main()
