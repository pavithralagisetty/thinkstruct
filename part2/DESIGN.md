# Part 2: Patent claim search at 10 million patents

This describes how to run the Part 1 search engine (`src/patent_search`) on the full patent
database: about 10 million patents instead of 640. It is meant to be read by someone who will
build it. It picks simple, well known tools and says where they will hurt.

The proof of concept in this folder (`README.md`, `faiss_ann.py`, `results.md`) measured the
key choice, approximate vector search, on our real vectors. Numbers marked *measured* come
from there or from Part 1; numbers marked *estimate* are scaled up from our data or taken from
public cloud prices and should be checked before money is spent.

## 1. What changes at scale

Part 1 keeps everything in one SQLite file and one numpy matrix, loads it all into memory, and
compares the question with every claim. That is exact and fast at 10,411 claims (*measured*:
under 1 ms for the vector part, about 8 ms for a whole search). At 10M patents:

| | Part 1 (640 patents) | 10M patents (*estimate*) | How |
|---|---|---|---|
| Patents | 640 | 10,000,000 | |
| Claims indexed | 10,411 | ~163,000,000 | 16.3 claims per patent |
| Raw JSON | 15.7 MB | ~250 GB | 24.5 KB per patent |
| Text and metadata (`search.db`) | 28 MB | ~450 GB | 44 KB per patent, mostly descriptions |
| Claim vectors, float32 | 32 MB | ~500 GB | 163M x 768 x 4 bytes |
| Exact search, one question | 0.6 ms | ~25 s | grows in step with the claims (*measured* 30 ms at 200k) |
| Keyword scan (regex over every claim) | 30-43 ms | ~10 min | grows in step with the text |

So three things must change:

1. **Vectors** no longer fit in one machine's memory and cannot all be compared per question.
   They move to a vector database with an approximate (HNSW) index, split over several machines.
2. **Text and metadata** move from SQLite to Postgres, and keyword search uses an index
   instead of scanning every claim.
3. **Building the index** is no longer one command that takes a minute. It becomes a pipeline of
   batches that can fail, be retried and be tracked.

What does **not** change: the cleaning rules (`load.py`, `claims.py`), the claim and patent
texts that get embedded, the model (nomic, pinned revision), the score
(`0.8 x claim + 0.2 x patent`), the keyword rules (`keywords.py`), the CPC rules (`cpc.py`)
and the API shape. The workers and the search service import the existing package.

## 2. Components

```
               weekly USPTO files
                       |
                       v
  +-----------+   +------------------+   +------------------+
  | S3 bucket |-->| parse workers    |-->| Postgres         |<------+
  | raw files |   | (CPU, load.py,   |   | patents, claims, |       |
  +-----------+   |  claims.py)      |   | jobs, batches    |       |
                  +------------------+   +------------------+       |
                                               |  claims to embed   |
                                               v                    |
                                         +------------------+       |
                                         | embed workers    |       |
                                         | (GPU, nomic)     |       |
                                         +------------------+       |
                                               |  vectors           |
                                               v                    |
  user --> Search API (FastAPI + nomic on CPU) --> Qdrant cluster   |
                 |        HNSW, int8 in RAM, filters                |
                 +--------------------------------------------------+
                          text, keyword candidates, display fields

  Dashboard + alerts: reads the jobs and batches tables and the API metrics
```

| Component | Choice | Why this one | Holds |
|---|---|---|---|
| Raw storage | S3 (or any object store) | Cheap, the files never change, we can always re-run from them | Source files exactly as downloaded, plus their SHA-256 |
| Database | Postgres 16 | One familiar database for text, metadata, keyword search and the job queue | `patents`, `claims` (same columns as `search.db`), `batches`, `jobs`, `ingest_warnings` |
| Job queue | A `jobs` table in Postgres, claimed with `SELECT ... FOR UPDATE SKIP LOCKED` | No extra system to run; jobs and data live in the same place | One row per job, with state, attempts, lease time, error |
| Parse workers | Python containers on CPU, importing `patent_search` | Reuse the Part 1 cleaning exactly | No state |
| Embed workers | Python containers on spot GPUs (one L4 or A10G each) | Embedding is the only heavy compute; spot is cheap and a lost job is just retried | No state |
| Vector database | Qdrant cluster | HNSW plus filters applied inside the search (the proof of concept showed filtering afterwards fails), int8 quantization, sharding and replicas built in, and it falls back to exact scoring when a filter is narrow | Claim vectors with a small payload; patent vectors in a second collection |
| Search API | The Part 1 FastAPI app, with `VectorStore` swapped for a Qdrant client | `VectorStore` was kept as the only code that touches vectors for exactly this | The nomic model for embedding questions |
| Dashboard | Grafana (or a small page) on the Postgres tables and API metrics | Status is already in the tables | Nothing |

**Qdrant payload per claim:** `claim_id`, `doc_number`, `cpc_subclass`, `cpc_group`,
`cpc_subgroup`, `pub_date` (as an integer `YYYYMMDD`), `is_independent`. Each of these gets a
payload index so filters are fast. The point id is a fixed hash of `claim_id`, so writing the
same claim twice replaces it instead of adding a copy.

**Memory in Qdrant** (*estimate*, from the proof of concept's 271 bytes of graph per vector at M=32):

| Part | Size |
|---|---|
| Claim vectors as int8, in RAM | 163M x 768 bytes = ~125 GB |
| HNSW graph, in RAM | 163M x ~271 bytes = ~44 GB |
| Patent vectors as int8, in RAM | 10M x 768 bytes = ~8 GB |
| Full float32 vectors, on disk (for re-scoring) | ~530 GB |

About 180 GB of RAM in total, split over **4 shards** (about 45 GB each), with **2 copies** of
each shard so one machine can fail: 8 machines with 64 GB RAM each.

## 3. Pipelines

### 3.1 Ingest (new data every week)

1. **Fetch.** A scheduled job downloads the week's files to `s3://.../raw/<date>/` and writes a
   `batches` row per file: `batch_id, source_file, sha256, state='fetched'`. A file whose
   SHA-256 is already in `batches` is skipped, so re-downloading is harmless.
2. **Parse.** One `parse` job per batch. The worker reads the file and runs the Part 1 rules:
   empty values, blank description items, cancelled claims, claim number recovery, parents and
   roots, spacing cleanup, CPC split. It writes `patents` and `claims` rows in **one
   transaction per batch**, so a crash leaves nothing half written. Warnings that the build
   logs today (numbering gaps, self references, unknown CPC codes) go into `ingest_warnings`
   with the `doc_number`. State becomes `parsed`.
3. **Duplicates.** Before writing, the worker looks up each `doc_number`. If it already exists
   with a later `pub_date`, the new record is skipped. If the new one is later (an A9 correcting
   an A1), the old patent's claims are deleted from Postgres and Qdrant and replaced. This is
   the Part 1 rule, now checked across batches instead of within one build.
4. **Embed.** One `embed` job per batch (about 1,000 to 10,000 claims). The worker reads the
   batch's `embed_text` and the patents' `{title}. {abstract}`, embeds them with the pinned
   nomic revision (`search_document: ` prefix) and upserts them into Qdrant with their payload.
5. **Done.** Only when Postgres and Qdrant both have the batch is it marked `indexed`, with its
   counts (patents, claims, cancelled claim numbers, claims with missing openings) and the model
   version.

### 3.2 Search

The API receives the same request as today. A small **query planner** chooses how to run it:

```
filters  = classification, dates, main claims only   -> Qdrant payload filter
keywords = claim keywords, title or abstract keywords -> Postgres full-text candidates

1. If there are keywords:
       candidates = Postgres full-text search (fast, may over-match)
       candidates = exact keyword regex from keywords.py on those texts only (exact rules kept)
       if no question: sort by doc_number, claim_number, return top_n. Done.
2. If there is no question (filters only):
       Postgres query with the filters, ORDER BY doc_number, claim_number LIMIT top_n. Done.
3. Embed the question (nomic on CPU, ~13 ms measured).
4. Ask Qdrant for the top 100 claims, with the filters (and the keyword candidate ids, if
   any) applied inside the search. When the filter leaves few claims, Qdrant scores them
   exactly instead of walking the graph (its full_scan_threshold), which is the
   "narrow filter -> exact" rule from the proof of concept.
5. Re-score those 100: fetch their full vectors and their patents' vectors and compute
   0.8 x claim + 0.2 x patent, round to 6 decimals, sort with the Part 1 tie order.
6. Fetch display fields from Postgres for the top_n, add highlights, group by patent if asked.
```

Why 100 candidates: the proof of concept found that 100 HNSW candidates re-scored with the
production score keep **96.3%** of the exact top 10 (*measured*, 0.41 ms), and 200 keep 97.2%.
`TOP_N_MAX` stays 100.

**Keyword search** uses a Postgres `tsvector` column on `claims.text` with the `simple`
configuration (no stemming, matching the Part 1 rule) and a GIN index. Phrases become `<->`
queries and `spoke*` becomes `spoke:*`. This only narrows the candidates; the final decision is
still the exact regex from `keywords.py`, so results are the same as Part 1. The hyphen rule
(`fiber reinforced` = `fiber-reinforced`) must be checked against how the `simple` parser splits
hyphenated words, and both forms put into the query if needed.

### 3.3 Re-embedding (model change)

Today a model change means "rebuild". At scale it means about a day of GPU time, so:

1. Create a new Qdrant collection, for example `claims_nomic_v2`.
2. Run `embed` jobs for every batch into it, while search keeps using the old one.
3. Run the evaluation (`python -m patent_search eval` and `grade score`) against it.
4. Switch the Qdrant alias `claims_active` to the new collection. Rolling back means pointing the
   alias back. Delete the old collection a week later.

Each collection records its model name, revision and code revision, and the API refuses to
start if its config does not match, the same check as `IndexNotReady` today.

## 4. Handling errors

| What fails | What happens |
|---|---|
| A worker crashes or a spot GPU is taken away | The job's lease (for example 15 minutes) runs out and another worker picks it up. Jobs are safe to repeat: Postgres writes are per batch transactions and Qdrant writes are upserts by a fixed id. |
| A job keeps failing | After 3 attempts it becomes `failed` with the error text, and the dashboard shows it. The rest of the pipeline continues. Someone fixes the cause and resets it to `pending`. |
| One bad patent record in a file | The record is skipped and written to `ingest_warnings`; the batch continues. Never fill in missing text (Part 1 rule). |
| Embedding runs out of GPU memory | Retry the job with half the batch size. |
| Postgres has a batch but Qdrant does not (or the other way round) | The batch is not `indexed`. A nightly check compares the claim count per batch in both systems and re-queues the `embed` job when they differ. |
| A new file format breaks claim number recovery | The warnings per batch jump. Alert when a batch has far more warnings or missing openings than usual (today about 30% of claim items have missing openings). |
| Qdrant node down | The second copy of the shard answers. Without replicas, search fails for that shard's claims, so replicas are worth their cost. |
| Search API overloaded | Add API instances; they hold no state except the model. |

## 5. Tracking contents and status

* **`batches` table** is the record of what is in the system: one row per source file with its
  SHA-256, state (`fetched`, `parsed`, `indexed`, `failed`), counts, model version and times.
  "What is searchable?" is `SELECT ... WHERE state = 'indexed'`.
* **`jobs` table** is the live work: state, attempts, worker, lease time, last error.
* **Dashboard:**
  * batches per state, and the oldest batch not yet indexed
  * patents and claims indexed, added this week
  * failed jobs, with their errors
  * warnings per batch
  * search p50 and p95 latency, error rate, requests per minute
  * Qdrant memory per node
* **Accuracy check (nightly).** Run the 40 evaluation questions on a sample shard with Qdrant
  and with exact scoring, and alert if recall@10 drops below about 0.9. This catches an index
  that was built with bad settings or is degrading, which plain uptime checks miss.
* **Quality check (on model or setting changes).** The Part 1 graded evaluation
  (`tests/eval/questions.json`, `grades/`) is rerun, with new pooled claims graded.

## 6. Cost

All **estimates**, on-demand AWS prices in us-east-1 (late 2026), rounded. Check them first.

**One time: embedding everything.**

* Text: 163M claims x ~210 tokens + 10M patents x ~175 tokens = about **36 billion tokens**
  (our claims average 871 characters of embedding text, about 4.2 characters per token).
* nomic (137M parameters) on one L4 GPU: roughly 1,500 to 3,000 claims per second, so about
  15 to 30 GPU hours; allow 60 with retries. At about $0.80 per GPU hour: **$20 to $100**.
* For comparison, a paid API (OpenAI `text-embedding-3-small`, $0.02 per million tokens) would
  be about $720, and every re-embedding costs that again.
* Parsing 10M patents is light CPU work: a few hours on a handful of workers, a few dollars.

**Monthly.**

| Part | Size | Cost per month |
|---|---|---|
| Qdrant: 8 machines, 64 GB RAM each (4 shards x 2 copies) plus ~150 GB disk each | r6i.2xlarge + gp3 | ~$3,050 |
| Postgres: 64 GB RAM, 1 TB disk | RDS db.r6i.2xlarge + gp3 | ~$850 |
| Search API: 2 instances (nomic runs on CPU) | c6i.xlarge | ~$250 |
| S3: ~250 GB raw files | | ~$10 |
| Weekly ingest (new patents only, spot GPU) | a few GPU hours | ~$20 |
| **Total** | | **~$4,200** |

Cheaper options: one copy of each Qdrant shard (about $1,500 less, but a machine failure takes
part of the index offline), or IVF-PQ with re-scoring instead of HNSW, which in the proof of
concept needed only 48 bytes per vector in RAM (8 GB for all claims) but kept less of the top 10
(0.96 on real data, 0.71 on the synthetic set) and needs fast disk for the re-score.

## 7. Challenges

* **Filters and approximate search do not mix well.** *Measured*: dropping results that fail a
  filter after an HNSW search left 1 to 3 of 10 results, and filtering inside HNSW still missed
  28 to 34% of the true top 10 at efSearch=512. The planner relies on Qdrant switching to exact
  scoring for narrow filters, and the threshold has to be tuned. Middle-sized filters (a few
  percent of 163M claims, millions of claims) are the hard case: too many to score exactly,
  too few for the graph to work well.
* **Keyword candidates can be huge.** `tire*` matches a third of our claims. Passing millions of
  ids from Postgres to Qdrant as a filter is slow. Large keyword sets need either keyword fields
  in the Qdrant payload (full-text payload index) or a search engine that does both (OpenSearch,
  Vespa). Start with Postgres, measure, change if needed.
* **Two stores can disagree.** Postgres is the source of truth. Qdrant can be rebuilt from it,
  and the nightly count check finds gaps, but for a short time after a write a claim can be in
  one and not the other.
* **Re-embedding is slow.** A model change means a day of GPU time and a second full collection
  (double memory during the switch). The evaluation must say the new model is clearly better
  first (the Part 1 rule).
* **Memory grows.** Every week adds patents; HNSW memory grows with them. Plan to add shards,
  and watch Qdrant memory on the dashboard.
* **Data changes after the fact.** Corrections (A9), reclassification (CPC codes change), and
  withdrawn applications all need updates or deletes, not only inserts.
* **New sources break the cleaning rules.** The claim recovery rules were written for this data
  (`N .`, `N :`, `N )` numbering, nested claims losing their openings). Other years or offices
  will have new patterns. Watch the warnings per batch.
* **Approximate means ranking can shift.** Part 1 promises "same input, same output". HNSW is
  deterministic for a fixed index, but adding patents changes the graph, so results for the
  same question can change slightly from week to week even for old claims. Keyword-only and
  filter-only searches stay exact.

## 8. Build order

Each step is small enough to finish and test on its own, starting on the 640-patent data and
growing to a larger sample before the full set.

1. **Postgres schema.** `patents`, `claims` (same columns as `search.db`), `batches`, `jobs`,
   `ingest_warnings`. *Done when* the Part 1 build can write into Postgres and the counts match
   (640 patents, 10,411 claims, 119 without description, 3,200 with missing openings).
2. **Job queue and parse workers.** One `parse` job per file, `SKIP LOCKED`, leases, 3 attempts.
   *Done when* killing a worker mid-job still gives the same counts after a retry, and running
   all jobs twice changes nothing.
3. **Embed workers into Qdrant.** One collection, int8 quantization, payload indexes. *Done when*
   Qdrant has 10,411 points and exact search in Qdrant returns the same top 10 as Part 1 for the
   40 evaluation questions.
4. **Search API on Qdrant.** Replace `VectorStore` with a Qdrant client: top 100 candidates,
   then the 0.8/0.2 re-score. *Done when* the Part 1 "Done when" searches give the same results
   and the evaluation scores are within noise of Part 1.
5. **Keyword candidates from Postgres.** `tsvector` + GIN, then the exact regex. *Done when*
   keyword results are identical to Part 1 (`sipe`: same claims).
6. **Dashboard and alerts.** Batches by state, failed jobs, warnings, latency, the nightly
   accuracy check.
7. **Scale test.** A larger sample (for example 1M patents), sharded. Measure latency, memory
   and recall with filters, and tune the exact-scoring threshold, `efSearch` and shard count
   before the full load.
