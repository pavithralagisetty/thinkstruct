# Part 2 proof of concept: approximate vector search (FAISS)

The system design is in [`DESIGN.md`](DESIGN.md). This tests its key choice: at 10M patents
(~163M claims) comparing the question with every claim is too slow, so search needs an
approximate index (HNSW). How much faster is it, and what does it miss?

## Result

| Claims | Exact (Part 1 today) | HNSW | Faster by |
|---|---|---|---|
| 10,411 (real) | 0.58 ms | 0.21 ms, 93% recall | ~3x |
| 200,000 (synthetic) | 30 ms | 0.46 ms, 88% recall | ~65x |

Recall = share of the exact top 10 found. Higher `efSearch` raises it (99.5% on real data at
0.88 ms).

* **Exact search grows with the data, HNSW barely does.** At 163M claims exact search would
  take ~25 s per question. On today's data both are under 1 ms, so Part 1 stays exact.
* **Two-stage keeps the ranking.** Take 100 HNSW candidates and re-score them with the Part 1
  score (`0.8 x claim + 0.2 x patent`): 96% of the exact top 10 is kept.
* **Filters break approximate search.** Filtering after HNSW left 1 to 3 of 10 results;
  filtering inside HNSW still missed about a third. Exact scoring of the filtered claims was
  perfect and fastest for narrow filters (0.03 ms for 120 claims). So: narrow filter → exact,
  broad or no filter → HNSW plus re-score.
* **Compressed vectors (PQ) need re-scoring.** 48 bytes per vector instead of 3,072, but they find
  under half the top 10 unless their shortlist is re-scored with the full vectors (then 96% on real data).

All tables: [`results.md`](results.md).

## Limits

* The 200k set is the real vectors copied with random noise, so the trends hold but the exact
  numbers at that size do not.
* One laptop (Apple M3), one core per question, no sharding.

## Run

```
.venv/bin/pip install -e ".[poc]"              # adds faiss-cpu
.venv/bin/python part2/faiss_ann.py            # about 4 min, writes part2/results.md
.venv/bin/python part2/faiss_ann.py --scale 0  # real data only, about 30 s
```

Needs a built index (`python -m patent_search build`).
