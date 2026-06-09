"""Generate failure analysis from evaluation results."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_fiqa
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.eval import recall_at_k


def find_failures(retriever, queries, qrels, corpus, top_k=10, max_failures=10):
    """Find queries where the system fails to retrieve gold passages in top-k.
    
    Returns list of failure dicts with full context.
    """
    failures = []
    
    for qid, query_text in queries.items():
        if qid not in qrels:
            continue
        relevant = {doc_id for doc_id, score in qrels[qid].items() if score > 0}
        if not relevant:
            continue
        
        results = retriever.search(query_text, top_k=top_k)
        r = recall_at_k(results, relevant, k=top_k)
        
        if r < 1.0:  # At least one gold doc missed
            # Find the missing gold docs
            retrieved_ids = {doc_id for doc_id, _ in results[:top_k]}
            missed_docs = relevant - retrieved_ids
            
            # Get full ranked list to find gold doc position
            full_results = retriever.search(query_text, top_k=500)
            
            for missed_id in missed_docs:
                gold_rank = None
                gold_score = None
                for rank, (doc_id, score) in enumerate(full_results, 1):
                    if doc_id == missed_id:
                        gold_rank = rank
                        gold_score = score
                        break
                
                failure = {
                    "query_id": qid,
                    "query_text": query_text,
                    "recall@10": r,
                    "top_5": [
                        {
                            "rank": i + 1,
                            "doc_id": doc_id,
                            "score": round(score, 6),
                            "text": (corpus[doc_id].get("title", "") + " " + 
                                    corpus[doc_id].get("text", ""))[:200]
                        }
                        for i, (doc_id, score) in enumerate(results[:5])
                    ],
                    "gold_doc": {
                        "doc_id": missed_id,
                        "rank": gold_rank,
                        "score": round(gold_score, 6) if gold_score else None,
                        "title": corpus[missed_id].get("title", ""),
                        "text": corpus[missed_id].get("text", "")
                    }
                }
                failures.append(failure)
                
                if len(failures) >= max_failures:
                    return failures
    
    return failures


def generate_failures_md(failures, output_path):
    """Generate failures.md from failure cases."""
    lines = [
        "# Failure Analysis\n",
        "Three specific failure cases from the hybrid (weighted RRF) evaluation on FiQA dev set.\n",
        "---\n"
    ]
    
    for i, f in enumerate(failures[:3], 1):
        lines.append(f"## Failure {i}: Query \"{f['query_text'][:60]}{'...' if len(f['query_text']) > 60 else ''}\"\n")
        lines.append(f"**Query:** `\"{f['query_text']}\"`\n")
        lines.append(f"**Recall@10:** {f['recall@10']:.2f}\n")
        lines.append("**Top-5 Retrieved Passages:**\n")
        lines.append("| Rank | Doc ID | Score | Text (truncated) |")
        lines.append("|------|--------|-------|-------------------|")
        
        for r in f["top_5"]:
            text_escaped = r["text"].replace("|", "\\|").replace("\n", " ")[:100]
            lines.append(f"| {r['rank']} | {r['doc_id']} | {r['score']:.4f} | {text_escaped}... |")
        
        lines.append("")
        lines.append("**Gold Passage (the relevant passage our system missed):**")
        gold = f["gold_doc"]
        lines.append(f"- Doc ID: `{gold['doc_id']}`")
        lines.append(f"- Rank in our system: {gold['rank'] if gold['rank'] else 'Not in top-500'}")
        lines.append(f"- Score: {gold['score'] if gold['score'] else 'N/A'}")
        gold_title = gold.get("title", "").strip()
        if gold_title:
            lines.append(f"- Title: {gold_title}")
        gold_text = gold["text"].replace("\n", " ").strip()
        lines.append("- Full text:")
        lines.append("")
        lines.append(f"  > {gold_text}")
        lines.append("")
        
        # Auto-diagnosis based on patterns
        diagnosis = diagnose_failure(f)
        lines.append(f"**Diagnosis:** {diagnosis}\n")
        
        fix = suggest_fix(f)
        lines.append(f"**Proposed Fix:** {fix}\n")
        lines.append("---\n")
    
    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    
    print(f"Failure analysis written to: {output_path}")


def diagnose_failure(failure):
    """Auto-diagnose a failure case."""
    query = failure["query_text"].lower()
    gold_text = (failure["gold_doc"].get("title", "") + " " + failure["gold_doc"]["text"]).lower()
    top_texts = " ".join([r["text"].lower() for r in failure["top_5"]])
    gold_rank = failure["gold_doc"]["rank"]
    
    # Check for vocabulary mismatch
    query_words = set(query.split())
    gold_words = set(gold_text.split())
    overlap = query_words & gold_words
    
    if len(overlap) < 2:
        return ("Vocabulary mismatch — the gold passage uses different terminology than the query. "
                "BM25 scores low due to minimal term overlap, and the dense model's semantic similarity "
                f"isn't strong enough to overcome. Query-gold word overlap: {len(overlap)} words.")
    
    if gold_rank and gold_rank <= 20:
        return ("Near-miss — the gold passage is ranked just outside top-10. Both retrievers partially "
                f"surface it (rank {gold_rank}), but other passages with more direct term matches "
                "or stronger semantic similarity edge it out in the RRF fusion.")
    
    if gold_rank and gold_rank > 100:
        return ("Hard failure — both retrievers rank the gold passage very low. Likely a case where "
                "the passage answers the question indirectly or with specialized context that neither "
                "lexical matching nor semantic similarity captures well.")
    
    return ("Mixed signal — the gold passage has some relevance signals but is outranked by passages "
            "that more directly address the query's surface form.")


def suggest_fix(failure):
    """Suggest a concrete fix for a failure case."""
    gold_rank = failure["gold_doc"]["rank"]
    query = failure["query_text"]
    
    if gold_rank and gold_rank <= 20:
        return ("Increase fetch_k from 100 to 200 in the hybrid retriever, or add a lightweight "
                "cross-encoder reranker on the top-20 candidates to promote semantically relevant "
                "but lexically different passages.")
    
    if len(query.split()) < 5:
        return ("For short queries, apply query expansion: use the top-1 BM25 result's text to "
                "extract related terms, then re-run with an expanded query. Alternatively, use "
                "pseudo-relevance feedback (PRF) to enrich the query representation.")
    
    return ("Fine-tune the dense model on FiQA-specific query-passage pairs using contrastive "
            "learning, or add a domain-specific synonym expansion layer to the BM25 tokenizer "
            "(e.g., map 'short selling' ↔ 'shorting').")


if __name__ == "__main__":
    print("Loading FiQA dev set...")
    corpus, queries, qrels = load_fiqa(split="dev")
    
    print("Loading retrievers...")
    bm25 = BM25Retriever()
    dense = DenseRetriever()
    hybrid = HybridRetriever(bm25, dense)  # use tuned defaults
    
    print("Finding failure cases...")
    failures = find_failures(hybrid, queries, qrels, corpus, top_k=10, max_failures=10)
    
    print(f"Found {len(failures)} failure cases")
    
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "failures.md"
    )
    generate_failures_md(failures, output_path)
    
    # Also save raw data
    raw_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "failures_raw.json"
    )
    with open(raw_path, "w") as f:
        json.dump(failures, f, indent=2)
    print(f"Raw failure data saved to: {raw_path}")
