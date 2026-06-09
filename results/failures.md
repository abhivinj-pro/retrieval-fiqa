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

**Gold Passage (the relevant passage our system missed):**
- Doc ID: `100764`
- Rank in our system: 64
- Score: 0.006781
- Full text:

  > "You don't specify which country you are in, so my answers are more from a best practice view than a legal view.. I don't intend on using it for personal use, but I mean it's just as possible. This is a dangerous proposition.. You shouldn't co-mingle business expenses with personal expenses.  If there is a chance this will happen, then stop, make it so that it won't happen. The big danger is in being able to have traceability between what you are doing for the business, and what you are doing for yourself.  If you are using this as a ""staging"" account for investments, etc., are those investments for yourself?  Or for the business?  Is tax treatment on capital gains and/or dividends the same for personal and business in your jurisdiction?  If you buy a widget, is the widget an expense against business income?  Or is it an out of pocket expense for personal consumption?  The former reduces your taxable income, the latter does not. I don't see the benefit of a real business account because those have features specific to maybe corporations, LLC, and etc. -- nothing beneficial to a sole proprietor who has no reports/employees. The real benefit is that there is a clear delineation between business income/expenses and personal income/expenses. This account can also accept money and hold it from business transactions/sales, and possibly transfer some to the personal account if there's no need for reinvesting said amount/percentage. What you are looking for is a commonly called a current account, because it is used for current expenses.  If you are moving money out of the account to your personal account, that speaks to paying yourself, which has other implications as well. The safest/cleanest way to do this is to: While this may sound like overkill, it is the only way to guarantee that income/expenses are allocated to the correct entity (i.e. you, or your business). From a Canadian standpoint:"

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

**Gold Passage (the relevant passage our system missed):**
- Doc ID: `314352`
- Rank in our system: 167
- Score: 0.003256
- Full text:

  > If it makes your finances easier, why not? My wife and I had his/hers/our since before we were married. I also have an account to handle transactions for my rental property, and one extra for PayPal use. I was paranoid to give out a checking account number with authorization for a third party to debit it, so that account has a couple hundred dollars, maximum. All this is just to explain that your finances should be arranged to simplify your life and make you comfortable.

**Diagnosis:** Hard failure — both retrievers rank the gold passage very low. Likely a case where the passage answers the question indirectly or with specialized context that neither lexical matching nor semantic similarity captures well.

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

**Gold Passage (the relevant passage our system missed):**
- Doc ID: `146317`
- Rank in our system: 83
- Score: 0.006087
- Full text:

  > "When I was younger I had a problem with Washington Mutual.  Someone had deposited a check in to my account then ran my account negative with a ""dupe"" of my debit card.  WaMu tied up my account for three months while they investigated because it wasn't simply a debit card fraud issue, this was check fraud (so they claimed).  At the time all the money I had in the world was in that account and the ordeal was extremely disruptive to my life.  Since the, I never spend on my debit card(s) and I keep more than one checking account to disperse the risk and avoid disruption in the event anything ever happens again. Now one of the accounts contains just enough money (plus a small buffer) to pay my general monthly expenses and the other is my actual checking account.   There's no harm in having more than one checking account and if you think it will enhance your finances, do it. Though, there's no reason to get a business account unless you've actually formed a business."

**Diagnosis:** Mixed signal — the gold passage has some relevance signals but is outranked by passages that more directly address the query's surface form.

**Proposed Fix:** Fine-tune the dense model on FiQA-specific query-passage pairs using contrastive learning, or add a domain-specific synonym expansion layer to the BM25 tokenizer (e.g., map 'short selling' ↔ 'shorting').

---
