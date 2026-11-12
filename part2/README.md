# Part 2 proof of concept: approximate vector search (FAISS)

The system design for 10M patents is in [`DESIGN.md`](DESIGN.md). This file is the proof of concept for its key choice.

## In short

* **Is the current search bad?** No. Part 1 compares the question with every claim, so it
  always returns the true top 10 by similarity, in under 1 ms on 10,411 claims. Its result
  quality is measured separately by the graded evaluation (nomic: Hit@5 0.925, nDCG@10 0.676,
  well above BM25's 0.538).
* **Why change it at scale?** Comparing with every claim gets slower in step with the number
  of claims. At 10M patents (about 160M claims) that is roughly 25 s per question and 490 GB of
  vectors (160M x 768 numbers x 4 bytes), too slow and too big for one machine.
* **Is approximate search faster?** Yes, and the gap grows with the data:

  | Claims | Exact (today) | HNSW, efSearch=64 | Faster by |
  |---|---|---|---|
  | 10,411 (real) | 0.58 ms | 0.21 ms (93% recall) | about 3x |
  | 200,000 (synthetic) | 30 ms | 0.46 ms (88% recall) | about 65x |
  | 160M (estimate) | about 25 s | a few ms | thousands of times |

  On today's data the saving (0.4 ms) does not matter: a whole search takes about 8 ms in
  Python plus 13 ms to embed the question. So Part 1 stays exact.
* **What it costs.** It misses some results (88 to 99% of the true top 10, set by
  `efSearch`), it must be built (105 s for 200k vectors), it needs about as much memory as the
  raw vectors, and it handles filters badly. With a narrow filter, exact scoring of the claims
  that pass is both faster and perfect.
* **The design that follows:** narrow filter → exact scoring of the matching claims (what Part 1
  does now); broad or no filter → HNSW for about 100 candidates, then the exact production score
  on those (keeps 96% of today's top 10).

## What was tested

The script puts our real nomic claim vectors (`data/index/claim_vectors.npy`) into FAISS and
queries it with the 40 evaluation questions (`tests/eval/questions.json`):

1. **Index types on real data.** Exact (numpy, as `VectorStore` does today, and FAISS
   `IndexFlatIP`), HNSW (a graph of nearest neighbours, M=32), int8 (each number stored in one
   byte, still a full scan) and IVF-PQ (claims grouped into lists, each vector squeezed into
   48 bytes), the last also with an exact re-score of its shortlist.
2. **Two-stage.** The production score is `0.8 x claim + 0.2 x patent` similarity; an ANN
   index only holds claim vectors. HNSW fetches K candidates and the full score ranks them.
   Compared with the production top 10 over all claims.
3. **Filters.** CPC `B60C 11` (16% of claims) and main claims only (1.2%): dropping failing hits
   after an HNSW search, filtering inside HNSW (FAISS `IDSelectorBatch`), and exact scoring of
   the claims that pass (Part 1 today).
4. **Scale.** The same index types on 200,000 synthetic vectors (see Limits).

## Run

```
.venv/bin/pip install -e ".[poc]"                  # adds faiss-cpu
.venv/bin/python part2/faiss_ann.py                # real data + 200k synthetic vectors, about 4 min
.venv/bin/python part2/faiss_ann.py --scale 0      # real data only, about 30 s
```

Options: `--scale N` synthetic vectors (0 skips), `--sigma` noise added to them (0.03),
`--repeats` timed rounds per question (5).

It needs a built `data/index` (`python -m patent_search build`). The first run embeds the 40
questions in a separate process (torch and FAISS each load their own OpenMP and crash together
in one process) and caches them in `part2/query_vectors.npy` (git ignored). All tables are
written to `part2/results.md`.

**Recall@10** = share of the exact top 10 (by claim similarity) that the method returns
(1.0 = same as today). Latency is one question at a time on one core, after a warm-up.

## Files

* `DESIGN.md`: the system design for 10M patents (components, pipelines, errors, status, cost, challenges, build order).
* `faiss_ann.py`: the script (quick proof-of-concept code, not part of the package).
* `results.md`: every table from the last run.
* `README.md`: this file.

## Results

Apple M3, 16 GB, FAISS 1.16.0. Full tables: `results.md`.

**Real claim vectors (10,411)**

| Method | Recall@10 | Median ms | Bytes / vector |
|---|---|---|---|
| Exact, numpy (today) | 1.000 | 0.58 | 3,072 |
| HNSW M=32, efSearch=64 | 0.932 | 0.21 | 3,343 |
| HNSW M=32, efSearch=128 | 0.973 | 0.41 | 3,343 |
| HNSW M=32, efSearch=256 | 0.995 | 0.88 | 3,343 |
| int8, exact scan | 0.993 | 5.15 | 768 |
| IVF-PQ (48 bytes) | 0.46 | 0.12 | 212 |
| IVF-PQ + exact re-score of 16 x 10 | 0.963 | 0.20 | 3,284 |

**Synthetic (200,000 = real vectors + noise)**

| Method | Recall@10 | Median ms | Build s |
|---|---|---|---|
| Exact, numpy (today) | 1.000 | 30.5 | 0 |
| HNSW, efSearch=64 | 0.875 | 0.46 | 105 |
| HNSW, efSearch=256 | 0.960 | 1.02 | 105 |
| IVF-PQ + exact re-score of 16 x 10 | 0.705 | 0.33 | 14 |

**Two-stage (real data).** The production score is `0.8 x claim + 0.2 x patent`, and an ANN index
only holds claim vectors. Fetching 100 HNSW candidates and computing the full score on them
keeps 96.3% of the production top 10 (0.41 ms); 200 candidates keep 97.2%.

**Filters (real data).** Truth is the exact top 10 among claims that pass the filter.

| Filter | Method | Hits returned (of 10) | Recall@10 | ms |
|---|---|---|---|---|
| CPC B60C 11 (16% of claims) | HNSW, then drop failing hits (k=100) | 2.9 | 0.28 | 0.29 |
| | HNSW with the filter inside (IDSelector, ef=512) | 9.9 | 0.72 | 1.42 |
| | Exact on the filtered claims (today) | 10.0 | 1.00 | 0.26 |
| Main claims only (1.2%) | HNSW, then drop failing hits (k=100) | 1.1 | 0.11 | 0.33 |
| | HNSW with the filter inside (IDSelector, ef=512) | 9.5 | 0.66 | 1.53 |
| | Exact on the filtered claims (today) | 10.0 | 1.00 | 0.03 |

## What it shows

* **Exact search grows linearly.** 0.58 ms at 10k, 30 ms at 200k (52x the time for 19x the
  vectors, once the matrix no longer fits in cache). At 160M claims that is roughly 25 s per
  question on one machine, plus 490 GB of RAM. Not an option.
* **HNSW barely grows.** 0.21 to 0.46 ms from 10k to 200k at efSearch=64 (graph search is
  roughly logarithmic). `efSearch` is the speed/recall knob: 256 gives 0.96 to 0.995 recall in
  about 1 ms. The costs are memory (it keeps the full vectors plus the graph, ~3.3 KB per claim,
  about 535 GB at 160M claims, so the index must be sharded) and build time (105 s for 200k on
  8 laptop cores, so about a day for 160M on one such machine, likely more since it grows faster than linear; it parallelizes across shards).
* **Compression alone loses too much.** 48-byte PQ codes are 64x smaller but find only 14 to 46% of
  the top 10. They work as a shortlist: re-scoring 16 x 10 candidates with the full vectors
  brings real-data recall back to 0.96. The full vectors can then live on disk or in a
  separate store, with only the PQ codes in RAM (160M x 48 B = 7.7 GB). int8 keeps 99% recall
  at a quarter of the memory, but it is still a full scan.
* **Two-stage keeps the ranking.** ANN for ~100 candidates, then the exact production score
  (with the patent term), matches today's top 10 96% of the time. This is the design for scale.
* **Filters break naive ANN.** Dropping hits that fail the filter after the search leaves 1 to 3
  of 10 results. Filtering inside the graph is better but still misses a third of the true
  top 10, and gets worse the narrower the filter. Exact scoring of the filtered claims is both
  perfect and fastest when the filter is narrow (0.03 ms for 120 claims). So the query planner
  in the design picks by filter size: **narrow filter → exact on the matching rows; broad or no
  filter → ANN (with filter inside) plus re-score.** The cut-off is wherever the number of
  passing claims gets too large to scan, about 1 to 5M vectors per shard (tens of ms).

## Limits

* The 200k set is synthetic: 10,411 real vectors, each copied about 19 times with random noise.
  Its neighbourhoods are clusters of near-copies, unlike real patents. That is probably why PQ
  scores so much lower on it (the noise is exactly the fine detail PQ throws away). The trends
  (exact grows linearly, HNSW does not, PQ needs re-scoring) are reliable; the exact recall
  numbers at scale are not.
* Recall is measured against the exact claim-similarity top 10, not against relevance grades.
  An ANN miss only matters if the missed claim was relevant.
* One machine, one core per query, no sharding, no concurrent load, index in RAM only.
* FAISS's 8-bit PQ (`PQ96`) took minutes to train on 10k vectors on this Mac, so the 4-bit
  fast-scan version (`PQ96x4fs`) is used instead.
