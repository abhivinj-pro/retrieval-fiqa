"""Unit tests for retrieval system components."""
import pytest
import numpy as np
from src.index.bm25_index import WordTokenizer, tokenize
from src.retrieval.hybrid_retriever import reciprocal_rank_fusion
from src.eval import recall_at_k, mrr, count_tokens


@pytest.fixture(scope="module")
def word_tokenizer():
    """Default word tokenizer (stopwords + Porter stemmer)."""
    return WordTokenizer()


class TestWordTokenizer:
    """Tests for the default word tokenizer."""

    def test_returns_non_empty_tokens(self, word_tokenizer):
        tokens = tokenize("What is short selling?", word_tokenizer)
        assert len(tokens) > 0
        assert all(isinstance(t, str) and t for t in tokens)

    def test_lowercasing(self, word_tokenizer):
        assert tokenize("NYSE Trading", word_tokenizer) == tokenize(
            "nyse trading", word_tokenizer
        )

    def test_stopwords_removed(self, word_tokenizer):
        # "is" and "the" are in the stopword list; "selling"/"stock" remain.
        tokens = tokenize("What is the stock market?", word_tokenizer)
        assert "is" not in tokens
        assert "the" not in tokens
        # Content words survive (stemmed).
        assert any(t.startswith("stock") for t in tokens)
        assert any(t.startswith("market") for t in tokens) or "market" in tokens

    def test_porter_stemming_collapses_morphology(self, word_tokenizer):
        # Porter collapses common inflectional morphology. "taxes" and
        # "taxed" both stem to "tax"; "running" -> "run"; "investments"
        # -> "invest". (Porter is intentionally conservative on -ation
        # forms, so "taxation" stems to "taxat" — not asserted here.)
        assert tokenize("taxes", word_tokenizer) == ["tax"]
        assert tokenize("taxed", word_tokenizer) == ["tax"]
        assert tokenize("running", word_tokenizer) == ["run"]
        assert tokenize("investments", word_tokenizer) == ["invest"]

    def test_currency_preserved(self, word_tokenizer):
        # "$89.5B" must survive intact as a single token, NOT be stemmed.
        tokens = tokenize("Apple reported $89.5B revenue.", word_tokenizer)
        assert "$89.5b" in tokens

    def test_percent_preserved(self, word_tokenizer):
        tokens = tokenize("Yields rose by 2.5% this quarter.", word_tokenizer)
        assert "2.5%" in tokens

    def test_ampersand_acronym_preserved(self, word_tokenizer):
        tokens = tokenize("The S&P 500 index gained.", word_tokenizer)
        assert "s&p" in tokens

    def test_hyphenated_preserved(self, word_tokenizer):
        tokens = tokenize("The 10-K filing was late.", word_tokenizer)
        assert "10-k" in tokens

    def test_deterministic(self, word_tokenizer):
        text = "The S&P 500 index gained 2.5% in 2023"
        assert tokenize(text, word_tokenizer) == tokenize(text, word_tokenizer)

    def test_empty_string(self, word_tokenizer):
        assert tokenize("", word_tokenizer) == []

    def test_to_from_dict_roundtrip(self):
        tok = WordTokenizer(use_stopwords=False, use_stemmer=False)
        restored = WordTokenizer.from_dict(tok.to_dict())
        assert restored.use_stopwords is False
        assert restored.use_stemmer is False
        # Without stopwords/stemmer, "is" survives and morphology is preserved.
        tokens = tokenize("This is taxes", restored)
        assert "is" in tokens
        assert "taxes" in tokens


class TestReciprocalRankFusion:
    """Tests for RRF scoring."""
    
    def test_single_list(self):
        results = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        fused = reciprocal_rank_fusion([results], k=60)
        # doc1 should be first (rank 1 -> score 1/(60+1))
        assert fused[0][0] == "doc1"
        assert fused[1][0] == "doc2"
        assert fused[2][0] == "doc3"
    
    def test_two_lists_agreement(self):
        """When both lists agree on top doc, it should have highest fused score."""
        list1 = [("docA", 0.9), ("docB", 0.5), ("docC", 0.3)]
        list2 = [("docA", 0.8), ("docC", 0.6), ("docB", 0.4)]
        
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        # docA is rank 1 in both -> highest score
        assert fused[0][0] == "docA"
    
    def test_rrf_scores_correct(self):
        """Verify exact RRF score computation."""
        list1 = [("doc1", 1.0), ("doc2", 0.5)]
        list2 = [("doc2", 1.0), ("doc1", 0.5)]
        
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        fused_dict = dict(fused)
        
        # doc1: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
        expected_doc1 = 1/61 + 1/62
        # doc2: 1/(60+2) + 1/(60+1) = 1/62 + 1/61
        expected_doc2 = 1/62 + 1/61
        
        assert abs(fused_dict["doc1"] - expected_doc1) < 1e-10
        assert abs(fused_dict["doc2"] - expected_doc2) < 1e-10
        # Scores should be equal (symmetric)
        assert abs(fused_dict["doc1"] - fused_dict["doc2"]) < 1e-10
    
    def test_k_parameter_effect(self):
        """Lower k gives more weight to top positions."""
        list1 = [("doc1", 1.0), ("doc2", 0.5)]
        
        fused_low_k = reciprocal_rank_fusion([list1], k=1)
        fused_high_k = reciprocal_rank_fusion([list1], k=100)
        
        # With low k, difference between rank 1 and 2 is larger
        low_k_ratio = fused_low_k[0][1] / fused_low_k[1][1]   # 1/2 vs 1/3 -> 1.5
        high_k_ratio = fused_high_k[0][1] / fused_high_k[1][1]  # 1/101 vs 1/102 -> ~1.01
        
        assert low_k_ratio > high_k_ratio
    
    def test_empty_lists(self):
        fused = reciprocal_rank_fusion([[]], k=60)
        assert fused == []
    
    def test_disjoint_lists(self):
        """Documents only in one list still get scored."""
        list1 = [("doc1", 1.0)]
        list2 = [("doc2", 1.0)]
        
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        assert len(fused) == 2
        # Both have same score (rank 1 in one list, absent from other)
        assert abs(fused[0][1] - fused[1][1]) < 1e-10

    def test_weighted_breaks_symmetric_tie(self):
        """Per-list weights are honored: heavier list wins disjoint ties."""
        list1 = [("docA", 1.0)]   # only in list1
        list2 = [("docB", 1.0)]   # only in list2
        fused = reciprocal_rank_fusion(
            [list1, list2], k=60, weights=[0.3, 0.7]
        )
        fused_dict = dict(fused)
        # docB carried by the heavier list ranks above docA.
        assert fused[0][0] == "docB"
        # Exact scores: docA = 0.3/(60+1), docB = 0.7/(60+1)
        assert abs(fused_dict["docA"] - 0.3 / 61) < 1e-12
        assert abs(fused_dict["docB"] - 0.7 / 61) < 1e-12

    def test_weighted_zero_silences_list(self):
        """A weight of 0 makes that list contribute nothing."""
        list1 = [("docA", 1.0)]
        list2 = [("docB", 1.0)]
        fused = reciprocal_rank_fusion(
            [list1, list2], k=60, weights=[0.0, 1.0]
        )
        ids = [d for d, _ in fused]
        assert ids == ["docB"]

    def test_weighted_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion(
                [[("d1", 1.0)], [("d2", 1.0)]], weights=[1.0]
            )


class TestEvalMetrics:
    """Tests for evaluation metrics."""
    
    def test_recall_at_k_perfect(self):
        results = [("doc1", 0.9), ("doc2", 0.8)]
        relevant = {"doc1", "doc2"}
        assert recall_at_k(results, relevant, k=10) == 1.0
    
    def test_recall_at_k_partial(self):
        results = [("doc1", 0.9), ("doc3", 0.8)]
        relevant = {"doc1", "doc2"}
        assert recall_at_k(results, relevant, k=10) == 0.5
    
    def test_recall_at_k_miss(self):
        results = [("doc3", 0.9), ("doc4", 0.8)]
        relevant = {"doc1", "doc2"}
        assert recall_at_k(results, relevant, k=10) == 0.0
    
    def test_recall_at_k_cutoff(self):
        """Only considers top-k results."""
        results = [("doc3", 0.9), ("doc1", 0.8)]
        relevant = {"doc1"}
        assert recall_at_k(results, relevant, k=1) == 0.0
        assert recall_at_k(results, relevant, k=2) == 1.0
    
    def test_recall_empty_relevant(self):
        results = [("doc1", 0.9)]
        assert recall_at_k(results, set(), k=10) == 0.0
    
    def test_mrr_first_position(self):
        results = [("doc1", 0.9), ("doc2", 0.8)]
        relevant = {"doc1"}
        assert mrr(results, relevant) == 1.0
    
    def test_mrr_second_position(self):
        results = [("doc2", 0.9), ("doc1", 0.8)]
        relevant = {"doc1"}
        assert mrr(results, relevant) == 0.5
    
    def test_mrr_not_found(self):
        results = [("doc2", 0.9), ("doc3", 0.8)]
        relevant = {"doc1"}
        assert mrr(results, relevant) == 0.0
    
    def test_count_tokens(self):
        assert count_tokens("hello world") == 2
        assert count_tokens("one two three four five") == 5
        assert count_tokens("") == 0  # empty string edge case might be 1 with split


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
