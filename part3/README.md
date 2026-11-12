# Part 3: making nomic better at patent claim search

Production search uses `nomic-embed-text-v1.5`. Part 3 tried four ways to adapt it to this data, and measured each one twice:

* **Its own held-out test:** the oldest 80% of weekly files train, the newest 20% (130 patents) are never trained on.
* **The real test:** the 40 evaluation questions, graded by pooled blind judges (1,267 graded claims, see `CLAUDE.md`, Evaluation).

## Result

| Model | MRR | nDCG@10 | P@5 | Full@10 | nDCG, 7 unseen-patent questions |
|---|---|---|---|---|---|
| nomic (production) | 0.857 | 0.672 | 0.770 | 34/40 | 0.805 |
| 1. Fine tune on abstracts → first claims | 0.799 | 0.633 | 0.750 | 29/40 | 0.648 |
| 2. Query adapter on abstracts | **0.904** | **0.701** | 0.790 | 33/40 | **0.844** |
| 3. Query adapter on summaries | 0.902 | 0.695 | **0.800** | 33/40 | 0.839 |
| 4. Fine tune on summaries → all claims | 0.824 | 0.647 | 0.770 | 28/40 | 0.714 |
| *For reference: qwen3-0.6b* | 0.898 | 0.708 | 0.805 | 35/40 | |

`Full@10` = questions with a fully relevant claim in the top 10. "Unseen-patent questions" are the 7 whose fully relevant claims are all in patents outside the training weeks (`q13 q19 q26 q29 q33 q35 q36`), the only questions that cannot reward memory.

* **The query adapter is the only change that helps and never hurts.** MRR +0.047 and nDCG +0.029 over nomic, and better on unseen patents too. The gains are within noise on 40 questions (paired bootstrap, 95% intervals include zero), so production stays on nomic by the rule "switch only on a clear win". It can be turned on with `PATENT_SEARCH_QUERY_ADAPTER=models/query_adapter.npz`, with no rebuild.
* **Full fine tunes learn their own task but get worse at search.** Both improved on their own test and both fell on the 40 questions, most on unseen patents. Retraining all of nomic's weights on ~500 patents overfits to them.
* **Better training data narrowed the gap but did not close it.** Fine tune 4 (short summaries as questions, the index's real claim texts as answers, hard wrong answers) beats fine tune 1 on every metric, but still sits below base nomic.
* **Summaries did not help the adapter.** Trained on abstracts or on summaries, it lands in the same place (top 10s share 8.5 of 10 claims).

## The four attempts

### 1. Fine tune: abstract → first claim (`training.py`, `part3_train_eval.py`)

One row per patent: question = abstract, right answer = first claim, wrong answer = another patent's first claim with the same 4 character classification. 510 training patents, MultipleNegativesRankingLoss, batch 4 on mps, 6 minutes.

| Own test (130 patents) | Base | Fine tuned |
|---|---|---|
| Recall@1 | 0.862 | 0.938 |
| Triplet accuracy | 0.962 | 0.985 |

**Why it failed at search:** it trained on long abstracts (median 125 words) as questions, while users type short sentences, and on bare first claims, while the index embeds `Patent / Main claim / Claim` texts of every claim. It learned "abstract → first claim", not search.

### 2. Query adapter on abstracts (`adapter.py`, `python -m patent_search adapter`)

A 768 × 768 linear layer on the question vector only: `q' = normalize(W q + b)`, starting as the identity (so untrained it changes nothing), held near it by a penalty on `||W − I||² + ||b||²`. Trained so each abstract scores its own patent's claims above every other patent's claims (softmax over all claims). Claim vectors and the index are untouched.

| Own test (130 patents) | Before | After |
|---|---|---|
| MRR@10 | 0.921 | 0.915 |
| Gap, own vs other claims' similarity | 0.211 | 0.333 |

Ranking on its own test barely moves (it was already high); on the search questions it is the best nomic variant. **Why it works:** it cannot drift far from nomic, so it cannot break what nomic already does well.

### 3. Query adapter on summaries (`adapter --source summary`)

Same layer, with a 1 to 2 sentence summary of each abstract as the question. The 640 summaries (`data/part3_summaries/summaries.json`, median 29 words) were written by subagents from the title and abstract only, without seeing the evaluation questions, then fact-checked by a second set of subagents against the same source: 128 were rewritten for stating something the abstract does not (mostly a vehicle type such as "bicycle" or "car", or an unstated purpose), 3 more fixed by hand.

Same result as attempt 2 (MRR −0.002, nDCG −0.006 against it). **Why:** a layer held near the identity learns roughly one shift of the question vector, and summaries and abstracts teach the same shift.

### 4. Fine tune: summary → every claim (`train_summary.py`)

Built to fix attempt 1:

* **Questions:** the summaries, with `search_query: `.
* **Right answers:** each claim's stored `embed_text` with `search_document: `, exactly as the index embeds it, dependent claims included; the main claim plus up to 4 others per patent (2,514 training rows).
* **Wrong answers:** a claim of another patent in the same CPC main group (`B60B 1`), else the same subclass, else any patent (train: 2,462 / 37 / 15 rows). A candidate is skipped if base nomic scores it at or above 90% of the right answer (probably also right) or its text is identical (5,110 and 1 skipped).
* **Batches** never hold two rows of the same patent, so a patent's own claim is never used as a wrong answer for it.

629 steps, 19 minutes on mps.

| Own test (130 patents) | Base | Fine tuned |
|---|---|---|
| nDCG@10 | 0.881 | 0.915 |
| Own claim in top 10 | 0.962 | 0.977 |

Better than attempt 1 on the search questions (nDCG +0.014, MRR +0.024), still below nomic (nDCG −0.025, MRR −0.034) and well below on unseen patents (0.805 → 0.714).

## Limits

* 40 questions and one judge per claim: every difference between nomic and the Part 3 models is within noise. The unseen-patent check rests on 7 questions.
* 30 of the evaluation questions were written from patents in the training weeks, so gains there could be memory; that is why the unseen-patent column is reported.
* Only pooled claims are graded; a relevant claim no candidate returned is invisible.
* One small dataset (640 patents, 10,411 claims, mostly B60B and B60C) and one 16 GB Mac (batch 4, so few in-batch wrong answers per step).

## What would be worth trying next

* More training patents before any full fine tune; the gap between "own test" and "search" says the model is overfitting to ~500 patents.
* Real or realistic search questions as training queries (the adapter would then learn the question → claim shift, not abstract → claim).
* Larger batches on a GPU for the fine tune, so each step sees more wrong answers.
* A second judge on a sample of grades, and more questions about unseen patents, to tell small gains from noise.

## Run

```
.venv/bin/python part3_train_eval.py --query_field abstract     # attempt 1 (about 6 min on mps)
.venv/bin/python -m patent_search adapter                       # attempt 2 (seconds)
.venv/bin/python -m patent_search adapter --source summary      # attempt 3
.venv/bin/python -m patent_search.train_summary --dry_run       # attempt 4: build rows, print counts
.venv/bin/python -m patent_search.train_summary                 # attempt 4: train (about 20 min on mps)
.venv/bin/python -m patent_search eval && .venv/bin/python -m patent_search grade prepare   # then blind judges, then grade score
```

Needs a built index (`python -m patent_search build`). Results per attempt: `results/part3_results.md`, `results/adapter_results.md`, `results/adapter_summary_results.md`, `results/part3_summary_results.md`.
