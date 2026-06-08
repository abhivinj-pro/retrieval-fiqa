# Failure Analysis

Three specific failure cases from the hybrid (weighted RRF) evaluation on FiQA dev set.

---

## Failure 1: Query "Having a separate bank account for business/investing, but n..."

**Query:** `"Having a separate bank account for business/investing, but not a “business account?”"`

**Recall@10:** 0.25

**Top-5 Retrieved Passages:**

| Rank | Doc ID | Score | Text (truncated) |
|------|--------|-------|-------------------|
| 1 | 64556 | 0.0161 |  If you're a sole proprietor there's no reason to have a separate business account, as long as you k... |
| 2 | 537593 | 0.0161 |  Yes, it's a good idea to have a separate business account for your business because it makes accoun... |
| 3 | 364378 | 0.0159 |  As an LLC you are required to have a separate bank account (so you can't have one account and mix p... |
| 4 | 296717 | 0.0151 |  "Having a separate checking account for the business makes sense. It simplifies documenting your in... |
| 5 | 109203 | 0.0151 |  You could, but the bank won't let you... If you're a sole proprietor - then you could probably open... |

**Gold Passage:**
- Doc ID: 146317
- Rank in our system: 83
- Score: 0.006087
- Text:  "When I was younger I had a problem with Washington Mutual.  Someone had deposited a check in to my account then ran my account negative with a ""dupe"" of my debit card.  WaMu tied up my account for...

**Diagnosis:** Mixed signal — the gold passage has some relevance signals but is outranked by passages that more directly address the query's surface form.

**Proposed Fix:** Fine-tune the dense model on FiQA-specific query-passage pairs using contrastive learning, or add a domain-specific synonym expansion layer to the BM25 tokenizer (e.g., map 'short selling' ↔ 'shorting').

---

## Failure 2: Query "Having a separate bank account for business/investing, but n..."

**Query:** `"Having a separate bank account for business/investing, but not a “business account?”"`

**Recall@10:** 0.25

**Top-5 Retrieved Passages:**

| Rank | Doc ID | Score | Text (truncated) |
|------|--------|-------|-------------------|
| 1 | 64556 | 0.0161 |  If you're a sole proprietor there's no reason to have a separate business account, as long as you k... |
| 2 | 537593 | 0.0161 |  Yes, it's a good idea to have a separate business account for your business because it makes accoun... |
| 3 | 364378 | 0.0159 |  As an LLC you are required to have a separate bank account (so you can't have one account and mix p... |
| 4 | 296717 | 0.0151 |  "Having a separate checking account for the business makes sense. It simplifies documenting your in... |
| 5 | 109203 | 0.0151 |  You could, but the bank won't let you... If you're a sole proprietor - then you could probably open... |

**Gold Passage:**
- Doc ID: 100764
- Rank in our system: 64
- Score: 0.006781
- Text:  "You don't specify which country you are in, so my answers are more from a best practice view than a legal view.. I don't intend on using it for personal use, but I mean it's just as possible. This i...

**Diagnosis:** Mixed signal — the gold passage has some relevance signals but is outranked by passages that more directly address the query's surface form.

**Proposed Fix:** Fine-tune the dense model on FiQA-specific query-passage pairs using contrastive learning, or add a domain-specific synonym expansion layer to the BM25 tokenizer (e.g., map 'short selling' ↔ 'shorting').

---

## Failure 3: Query "Having a separate bank account for business/investing, but n..."

**Query:** `"Having a separate bank account for business/investing, but not a “business account?”"`

**Recall@10:** 0.25

**Top-5 Retrieved Passages:**

| Rank | Doc ID | Score | Text (truncated) |
|------|--------|-------|-------------------|
| 1 | 64556 | 0.0161 |  If you're a sole proprietor there's no reason to have a separate business account, as long as you k... |
| 2 | 537593 | 0.0161 |  Yes, it's a good idea to have a separate business account for your business because it makes accoun... |
| 3 | 364378 | 0.0159 |  As an LLC you are required to have a separate bank account (so you can't have one account and mix p... |
| 4 | 296717 | 0.0151 |  "Having a separate checking account for the business makes sense. It simplifies documenting your in... |
| 5 | 109203 | 0.0151 |  You could, but the bank won't let you... If you're a sole proprietor - then you could probably open... |

**Gold Passage:**
- Doc ID: 314352
- Rank in our system: 167
- Score: 0.003256
- Text:  If it makes your finances easier, why not? My wife and I had his/hers/our since before we were married. I also have an account to handle transactions for my rental property, and one extra for PayPal ...

**Diagnosis:** Hard failure — both retrievers rank the gold passage very low. Likely a case where the passage answers the question indirectly or with specialized context that neither lexical matching nor semantic similarity captures well.

**Proposed Fix:** Fine-tune the dense model on FiQA-specific query-passage pairs using contrastive learning, or add a domain-specific synonym expansion layer to the BM25 tokenizer (e.g., map 'short selling' ↔ 'shorting').

---
