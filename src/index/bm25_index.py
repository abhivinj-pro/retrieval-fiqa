"""BM25 indexing over the FiQA corpus.

A single tokenizer is used: a Lucene-StandardAnalyzer-style English
pipeline (NFKC + lowercase, symbol-aware pre-tokenization that
preserves finance atoms like ``$1.2B`` / ``2.5%`` / ``10-K`` / ``s&p``,
stopword removal, Porter stemming). This is the configuration BM25's
scoring formula and the standard BEIR baselines were validated against.
"""
import json
import os
import pickle
import re
import unicodedata
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi


INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "index_artifacts")
WORD_TOKENIZER_FILENAME = "word_tokenizer.json"


# Lucene/BEIR-standard English stopwords. Small closed-class words whose
# IDF contribution would otherwise swamp short queries.
ENGLISH_STOPWORDS: frozenset = frozenset(
    [
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
        "if", "in", "into", "is", "it", "no", "not", "of", "on", "or",
        "such", "that", "the", "their", "then", "there", "these", "they",
        "this", "to", "was", "will", "with",
    ]
)

# Atomic tokens that must survive intact. Order matters: more specific
# patterns are listed first so the alternation prefers them over the
# generic ``[a-z0-9]+`` fallback. All patterns assume the input has
# already been lowercased.
_ATOM_PATTERN = re.compile(
    r"\$\d[\d,.]*[bmk]?"                  # currency: $1.2b, $500, $1,000.50
    r"|\d+(?:\.\d+)?%"                    # percent: 2.5%, 10%
    r"|[a-z]&[a-z](?:&?[a-z])*"           # ampersand acronyms: s&p, p&g, at&t
    r"|[a-z0-9]+(?:-[a-z0-9]+)+"          # hyphenated: 10-k, price-to-earnings
    r"|[a-z0-9]+"                         # word fallback
)


class WordTokenizer:
    """English word-level tokenizer with finance-symbol preservation.

    Pipeline: NFKC normalize -> lowercase -> regex pre-tokenize (atoms-aware)
    -> stopword removal -> Porter stem (alphabetic tokens only).
    """

    def __init__(self, use_stopwords: bool = True, use_stemmer: bool = True):
        self.use_stopwords = use_stopwords
        self.use_stemmer = use_stemmer
        self._stemmer = None
        if use_stemmer:
            # Lazy import keeps the nltk dependency localized.
            from nltk.stem import PorterStemmer

            self._stemmer = PorterStemmer()

    def encode(self, text: str) -> List[str]:
        if not text:
            return []
        text = unicodedata.normalize("NFKC", text).lower()
        tokens = _ATOM_PATTERN.findall(text)
        out: List[str] = []
        stopwords = ENGLISH_STOPWORDS if self.use_stopwords else frozenset()
        for tok in tokens:
            if tok in stopwords:
                continue
            # Only stem pure alphabetic words; symbol-bearing atoms must
            # round-trip unchanged so they match their indexed counterparts.
            if self._stemmer is not None and tok.isalpha():
                tok = self._stemmer.stem(tok)
            out.append(tok)
        return out

    def to_dict(self) -> dict:
        return {
            "use_stopwords": self.use_stopwords,
            "use_stemmer": self.use_stemmer,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WordTokenizer":
        return cls(
            use_stopwords=data.get("use_stopwords", True),
            use_stemmer=data.get("use_stemmer", True),
        )


def tokenize(text: str, tokenizer: WordTokenizer) -> List[str]:
    """Tokenize ``text`` with the given tokenizer instance."""
    if not text:
        return []
    return tokenizer.encode(text)


def build_bm25_index(
    doc_ids: List[str],
    texts: List[str],
    save_path: Optional[str] = None,
    tokenizer: Optional[WordTokenizer] = None,
) -> Tuple[BM25Okapi, List[str], WordTokenizer]:
    """Build BM25 index from documents.

    Args:
        doc_ids: document identifiers, parallel to ``texts``.
        texts: document strings.
        save_path: where to persist artifacts (default ``index_artifacts/bm25``).
        tokenizer: explicit tokenizer instance; defaults to ``WordTokenizer()``.

    Returns:
        bm25: BM25Okapi instance.
        doc_ids: same list passed in, returned for convenience.
        tokenizer: the tokenizer used (also persisted to disk).
    """
    if tokenizer is None:
        tokenizer = WordTokenizer()

    tokenized = [tokenize(t, tokenizer) for t in texts]
    bm25 = BM25Okapi(tokenized)

    if save_path is None:
        save_path = os.path.join(INDEX_DIR, "bm25")
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, WORD_TOKENIZER_FILENAME), "w") as f:
        json.dump(tokenizer.to_dict(), f)

    with open(os.path.join(save_path, "bm25_index.pkl"), "wb") as f:
        pickle.dump({"bm25": bm25, "doc_ids": doc_ids}, f)

    return bm25, doc_ids, tokenizer


def load_bm25_index(
    load_path: Optional[str] = None,
) -> Tuple[BM25Okapi, List[str], WordTokenizer]:
    """Load a pre-built BM25 index along with its tokenizer."""
    if load_path is None:
        load_path = os.path.join(INDEX_DIR, "bm25")

    with open(os.path.join(load_path, "bm25_index.pkl"), "rb") as f:
        data = pickle.load(f)

    cfg_path = os.path.join(load_path, WORD_TOKENIZER_FILENAME)
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        tokenizer = WordTokenizer.from_dict(cfg)
    else:
        tokenizer = WordTokenizer()

    return data["bm25"], data["doc_ids"], tokenizer
