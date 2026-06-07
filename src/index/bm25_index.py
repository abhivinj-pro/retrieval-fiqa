"""BM25 indexing and retrieval over the FiQA corpus.

Two tokenizers are supported and selectable at build time:

* ``"word"`` (default) — a Lucene-StandardAnalyzer-style English pipeline
  (NFKC + lowercase, symbol-aware pre-tokenization that preserves finance
  atoms like ``$1.2B`` / ``2.5%`` / ``10-K`` / ``s&p``, stopword removal,
  Porter stemming). This is the configuration BM25's scoring formula and
  the standard BEIR baselines were validated against.
* ``"bpe"`` — a byte-level BPE tokenizer trained on the corpus. Kept
  behind a flag so the trade-off can be ablated; documented to dilute IDF
  on rare finance terms but never produces ``<unk>``.

The tokenizer choice is persisted alongside the BM25 index so the
retriever loads the correct one transparently.
"""
import json
import os
import pickle
import re
import unicodedata
from typing import List, Optional, Tuple, Union

from rank_bm25 import BM25Okapi
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC, Lowercase, Sequence as NormalizerSequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "index_artifacts")
BPE_TOKENIZER_FILENAME = "bpe_tokenizer.json"
WORD_TOKENIZER_FILENAME = "word_tokenizer.json"
TOKENIZER_KIND_FILENAME = "tokenizer_kind.txt"

DEFAULT_TOKENIZER_KIND = "word"
DEFAULT_VOCAB_SIZE = 30000
DEFAULT_MIN_FREQUENCY = 2


# ---------------------------------------------------------------------------
# Word tokenizer (default)
# ---------------------------------------------------------------------------

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
            "kind": "word",
            "use_stopwords": self.use_stopwords,
            "use_stemmer": self.use_stemmer,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WordTokenizer":
        return cls(
            use_stopwords=data.get("use_stopwords", True),
            use_stemmer=data.get("use_stemmer", True),
        )


# ---------------------------------------------------------------------------
# BPE tokenizer (legacy, behind a flag)
# ---------------------------------------------------------------------------


def train_bpe_tokenizer(
    texts: List[str],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
) -> Tokenizer:
    """Train a byte-level BPE tokenizer on the given texts."""
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = NormalizerSequence([NFKC(), Lowercase()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<unk>"],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------

# Anything with ``.encode(str) -> List[str]`` (WordTokenizer) OR an HF
# ``Tokenizer`` (whose ``.encode`` returns an ``Encoding`` with ``.tokens``).
TokenizerLike = Union[WordTokenizer, Tokenizer]


def tokenize(text: str, tokenizer: TokenizerLike) -> List[str]:
    """Tokenize ``text`` with the given tokenizer instance."""
    if not text:
        return []
    if isinstance(tokenizer, WordTokenizer):
        return tokenizer.encode(text)
    return tokenizer.encode(text).tokens


# ---------------------------------------------------------------------------
# Build / load
# ---------------------------------------------------------------------------


def build_bm25_index(
    doc_ids: List[str],
    texts: List[str],
    save_path: Optional[str] = None,
    tokenizer: Optional[TokenizerLike] = None,
    tokenizer_kind: str = DEFAULT_TOKENIZER_KIND,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> Tuple[BM25Okapi, List[str], TokenizerLike]:
    """Build BM25 index from documents.

    Args:
        doc_ids: document identifiers, parallel to ``texts``.
        texts: document strings.
        save_path: where to persist artifacts (default ``index_artifacts/bm25``).
        tokenizer: explicit tokenizer instance. If supplied, the kind is
            inferred from its type and ``tokenizer_kind`` is ignored.
        tokenizer_kind: ``"word"`` (default) or ``"bpe"``. Only consulted
            when ``tokenizer`` is None.
        vocab_size: BPE vocab size; ignored for the word tokenizer.

    Returns:
        bm25: BM25Okapi instance.
        doc_ids: same list passed in, returned for convenience.
        tokenizer: the tokenizer used (also persisted to disk).
    """
    if tokenizer is None:
        if tokenizer_kind == "word":
            tokenizer = WordTokenizer()
        elif tokenizer_kind == "bpe":
            tokenizer = train_bpe_tokenizer(texts, vocab_size=vocab_size)
        else:
            raise ValueError(
                f"Unknown tokenizer_kind={tokenizer_kind!r}; expected 'word' or 'bpe'"
            )

    kind = "word" if isinstance(tokenizer, WordTokenizer) else "bpe"

    tokenized = [tokenize(t, tokenizer) for t in texts]
    bm25 = BM25Okapi(tokenized)

    if save_path is None:
        save_path = os.path.join(INDEX_DIR, "bm25")
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, TOKENIZER_KIND_FILENAME), "w") as f:
        f.write(kind)

    if isinstance(tokenizer, WordTokenizer):
        with open(os.path.join(save_path, WORD_TOKENIZER_FILENAME), "w") as f:
            json.dump(tokenizer.to_dict(), f)
    else:
        tokenizer.save(os.path.join(save_path, BPE_TOKENIZER_FILENAME))

    with open(os.path.join(save_path, "bm25_index.pkl"), "wb") as f:
        pickle.dump({"bm25": bm25, "doc_ids": doc_ids}, f)

    return bm25, doc_ids, tokenizer


def load_bm25_index(
    load_path: Optional[str] = None,
) -> Tuple[BM25Okapi, List[str], TokenizerLike]:
    """Load a pre-built BM25 index along with its tokenizer.

    Auto-detects which tokenizer was used at build time via the
    ``tokenizer_kind.txt`` marker, falling back to BPE for older artifacts
    that pre-date this file.
    """
    if load_path is None:
        load_path = os.path.join(INDEX_DIR, "bm25")

    with open(os.path.join(load_path, "bm25_index.pkl"), "rb") as f:
        data = pickle.load(f)

    kind_path = os.path.join(load_path, TOKENIZER_KIND_FILENAME)
    if os.path.exists(kind_path):
        with open(kind_path, "r") as f:
            kind = f.read().strip()
    else:
        kind = "bpe"  # legacy artifact

    if kind == "word":
        cfg_path = os.path.join(load_path, WORD_TOKENIZER_FILENAME)
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            tokenizer: TokenizerLike = WordTokenizer.from_dict(cfg)
        else:
            tokenizer = WordTokenizer()
    elif kind == "bpe":
        tokenizer = Tokenizer.from_file(os.path.join(load_path, BPE_TOKENIZER_FILENAME))
    else:
        raise ValueError(f"Unknown tokenizer kind {kind!r} in artifact at {load_path}")

    return data["bm25"], data["doc_ids"], tokenizer
