# Part 2: Patent claim search at 10 million patents

How to run the Part 1 search engine on the full patent database: about 10 million patents
instead of 640. Measured numbers come from Part 1 and the proof of concept in this folder
(`README.md`); everything else is an estimate scaled up from our data.

## 1. The problem

| | Part 1 (640 patents) | 10M patents |
|---|---|---|
| Claims | 10,411 | ~163 million |
| Claim vectors | 32 MB | ~500 GB |
| Exact search per question | 0.6 ms | ~25 s |

The vectors no longer fit in one machine's memory, and comparing the question with every
claim is too slow. The cleaning, keyword, CPC and scoring code from Part 1 stays the same;
only storage and search change.

## 2. Components

```
USPTO files -> S3 -> parse workers -> Postgres -> embed workers -> Qdrant
                                          ^                          ^
                                          +------ Search API --------+
```

| Component | Job |
|---|---|
| **S3** | Raw files as downloaded. Everything can be rebuilt from here. |
| **Postgres** | Patent and claim text and metadata (same columns as `search.db`), keyword search, and the job queue (a `jobs` table). |
| **Parse workers** (CPU) | Clean files with the Part 1 code (`load.py`, `claims.py`). |
| **Embed workers** (GPU) | Embed claims and patents with nomic. |
| **Qdrant** | Vector database: HNSW index with filters, 4 shards with 2 copies each. |
| **Search API** | The Part 1 FastAPI app, with `VectorStore` replaced by a Qdrant client. |
| **Dashboard** | Shows pipeline status and search health. |

## 3. Pipelines

**Ingest (weekly).**

1. Download new files to S3 and add one row per file to a `batches` table.
2. A parse worker cleans the file and writes its patents and claims to Postgres in one transaction.
3. An embed worker embeds the new claims and writes them to Qdrant, with the filter fields
   (CPC, date, main claim or not).
4. The batch is marked `indexed` only when both Postgres and Qdrant have it.

When the same patent arrives again (an A9 correcting an A1), the later one replaces the old
one, as in Part 1.

**Search.**

1. **Keywords:** Postgres full-text search finds candidate claims, then the exact Part 1
   keyword rules check only those.
2. **No question:** return the matching claims from Postgres. Done.
3. **Question:** embed it, and ask Qdrant for the top 100 claims with the filters applied
   inside the search. For narrow filters Qdrant scores the matching claims exactly.
4. Re-score those 100 with the Part 1 score (`0.8 x claim + 0.2 x patent`) and return the top `top_n`.

The proof of concept measured why: 100 approximate candidates, re-scored, keep 96% of the
exact top 10. Filtering after the search instead of inside it left only 1 to 3 of 10 results.

**New model.** Embed everything into a second Qdrant collection, run the evaluation on it,
then switch over. The old collection stays until the new one is confirmed.

## 4. Errors

* **Worker crash:** each job has a lease. When it runs out, another worker retries the job.
  Writes can safely repeat (one transaction per batch, upserts by claim id).
* **Repeated failure:** after 3 attempts the job is marked `failed` with its error and shown on
  the dashboard. The rest of the pipeline keeps going.
* **Bad record:** skip it and log a warning. Never fill in missing text.
* **Postgres and Qdrant out of sync:** a nightly check compares claim counts per batch and
  re-runs the embedding where they differ.
* **Machine down:** each Qdrant shard has a second copy.

## 5. Tracking

* `batches` table: every source file, its state (`fetched`, `parsed`, `indexed`, `failed`) and counts.
* `jobs` table: current work, attempts and errors.
* Dashboard: batches by state, failed jobs, claims indexed, search latency and errors.
* Nightly accuracy check: run the 40 evaluation questions with exact and approximate search,
  and alert if recall drops below 0.9.

## 6. Cost (estimates, AWS on-demand)

**One time:** embedding ~36 billion tokens (163M claims and 10M patents) takes about 15 to 60
GPU hours, **$20 to $100**. (A paid API like OpenAI would be about $720.)

**Monthly:**

| Part | Cost |
|---|---|
| Qdrant: 8 machines with 64 GB RAM (4 shards x 2 copies) | ~$3,050 |
| Postgres: 64 GB RAM, 1 TB disk | ~$850 |
| Search API: 2 servers | ~$250 |
| S3 and weekly ingest | ~$30 |
| **Total** | **~$4,200** |

Without the second copy of each shard: about $1,500 less, but one machine failure takes part
of the index offline.

## 7. Challenges

* **Filters with approximate search.** Medium-sized filters (millions of claims) are too many
  to score exactly and too few for the HNSW graph to work well.
* **Large keyword matches.** A common keyword like `tire*` matches a third of all claims, which
  is too many ids to pass from Postgres to Qdrant.
* **Two stores.** Postgres and Qdrant can briefly disagree. Postgres is the source of truth.
* **Re-embedding.** A model change costs a day of GPU time and double memory during the switch.
* **Growth and updates.** Memory grows every week, and corrections and reclassified patents
  need updates, not only inserts.
* **New data formats.** The claim cleaning rules fit this data. Other sources may break them;
  watch the warnings per batch.

## 8. Build order

1. Postgres tables; load the 640 patents and check the Part 1 counts match.
2. Job queue and parse workers, with retries.
3. Embed workers into Qdrant; check its top 10 matches Part 1 on the evaluation questions.
4. Search API on Qdrant, with the re-score.
5. Keyword candidates from Postgres; check results match Part 1 exactly.
6. Dashboard and alerts.
7. Test on about 1M patents and tune before the full load.
