# Part 2 proof of concept: approximate vector search with FAISS

FAISS 1.16.0, 40 evaluation questions embedded with nomic-ai/nomic-embed-text-v1.5, 768 dims, k = 10. Latency is one question at a time on one CPU core (FAISS) after a warm-up, 3 rounds. numpy exact uses Accelerate (all cores). Recall@10 = share of the exact top 10 (by claim similarity) that the method returns.

### Real claim vectors (10,411)

| Method | Setting | Build s | Size MB | Bytes / vector | Recall@10 | Median ms | p95 ms |
|---|---|---|---|---|---|---|---|
| numpy exact (production today) | - | 0 | 32 | 3072 | 1.000 | 0.58 | 0.68 |
| FAISS exact (IndexFlatIP) | - | 0.0 | 32 | 3072 | 1.000 | 1.09 | 1.17 |
| HNSW M=32 | efSearch=16 | 1.0 | 35 | 3343 | 0.760 | 0.08 | 0.13 |
| HNSW M=32 | efSearch=32 | 1.0 | 35 | 3343 | 0.863 | 0.12 | 0.21 |
| HNSW M=32 | efSearch=64 | 1.0 | 35 | 3343 | 0.932 | 0.21 | 0.32 |
| HNSW M=32 | efSearch=128 | 1.0 | 35 | 3343 | 0.973 | 0.41 | 0.65 |
| HNSW M=32 | efSearch=256 | 1.0 | 35 | 3343 | 0.995 | 0.88 | 1.56 |
| int8 exact (SQ8) | - | 0.2 | 8 | 768 | 0.993 | 5.15 | 8.82 |
| IVF-PQ 48 B (408 lists) | nprobe=4 | 1.3 | 2 | 212 | 0.485 | 0.11 | 0.16 |
| IVF-PQ 48 B (408 lists) | nprobe=16 | 1.3 | 2 | 212 | 0.460 | 0.12 | 0.17 |
| IVF-PQ 48 B (408 lists) | nprobe=64 | 1.3 | 2 | 212 | 0.455 | 0.15 | 0.21 |
| IVF-PQ 48 B (408 lists) + exact re-score | nprobe=16, k_factor=4 | 1.1 | 34 | 3284 | 0.835 | 0.14 | 0.17 |
| IVF-PQ 48 B (408 lists) + exact re-score | nprobe=16, k_factor=16 | 1.1 | 34 | 3284 | 0.963 | 0.20 | 0.23 |

### Two-stage: HNSW candidates, then the production score

Production score = 0.8 x claim similarity + 0.2 x patent similarity, over every claim. The ANN index only knows claim vectors, so it fetches K candidates and the full score is computed on those.

| Candidates K | efSearch | Overlap with production top 10 | Median ms |
|---|---|---|---|
| 10 | 64 | 0.890 | 0.22 |
| 50 | 64 | 0.935 | 0.25 |
| 100 | 100 | 0.963 | 0.41 |
| 200 | 200 | 0.972 | 0.84 |

### Filters with an ANN index

Truth is the exact top 10 among the claims that pass the filter. *Post-filter*: ask HNSW for k results, drop those that fail the filter. *In-index*: FAISS `IDSelectorBatch`, HNSW only returns rows that pass. *Exact on subset*: what production does today (filter first, score what passed).

| Filter | Claims passing | Method | Avg hits returned (of 10) | Recall@10 | Median ms |
|---|---|---|---|---|---|
| CPC B60C 11 | 1,658 (15.9%) | post-filter, k=10 | 0.8 | 0.075 | 0.22 |
| CPC B60C 11 | 1,658 (15.9%) | post-filter, k=100 | 2.9 | 0.280 | 0.29 |
| CPC B60C 11 | 1,658 (15.9%) | in-index (IDSelector), efSearch=64 | 7.7 | 0.330 | 0.22 |
| CPC B60C 11 | 1,658 (15.9%) | in-index (IDSelector), efSearch=512 | 9.9 | 0.718 | 1.42 |
| CPC B60C 11 | 1,658 (15.9%) | exact on subset | 10.0 | 1.000 | 0.26 |
| Main claims only | 120 (1.2%) | post-filter, k=10 | 0.1 | 0.015 | 0.19 |
| Main claims only | 120 (1.2%) | post-filter, k=100 | 1.1 | 0.113 | 0.33 |
| Main claims only | 120 (1.2%) | in-index (IDSelector), efSearch=64 | 3.3 | 0.173 | 0.21 |
| Main claims only | 120 (1.2%) | in-index (IDSelector), efSearch=512 | 9.5 | 0.655 | 1.53 |
| Main claims only | 120 (1.2%) | exact on subset | 10.0 | 1.000 | 0.03 |

### Synthetic vectors (200,000, real vectors + noise sigma=0.03)

| Method | Setting | Build s | Size MB | Bytes / vector | Recall@10 | Median ms | p95 ms |
|---|---|---|---|---|---|---|---|
| numpy exact (production today) | - | 0 | 614 | 3072 | 1.000 | 30.45 | 46.29 |
| FAISS exact (IndexFlatIP) | - | 0.4 | 614 | 3072 | 1.000 | 32.21 | 37.46 |
| HNSW M=32 | efSearch=16 | 105.4 | 669 | 3344 | 0.698 | 0.22 | 0.41 |
| HNSW M=32 | efSearch=32 | 105.4 | 669 | 3344 | 0.783 | 0.32 | 0.78 |
| HNSW M=32 | efSearch=64 | 105.4 | 669 | 3344 | 0.875 | 0.46 | 0.98 |
| HNSW M=32 | efSearch=128 | 105.4 | 669 | 3344 | 0.902 | 0.62 | 1.42 |
| HNSW M=32 | efSearch=256 | 105.4 | 669 | 3344 | 0.960 | 1.02 | 2.38 |
| int8 exact (SQ8) | - | 0.5 | 154 | 768 | 0.988 | 49.25 | 85.44 |
| IVF-PQ 48 B (1788 lists) | nprobe=4 | 22.0 | 18 | 90 | 0.158 | 0.33 | 0.86 |
| IVF-PQ 48 B (1788 lists) | nprobe=16 | 22.0 | 18 | 90 | 0.143 | 0.30 | 0.41 |
| IVF-PQ 48 B (1788 lists) | nprobe=64 | 22.0 | 18 | 90 | 0.138 | 0.41 | 0.49 |
| IVF-PQ 48 B (1788 lists) + exact re-score | nprobe=16, k_factor=4 | 13.7 | 633 | 3162 | 0.365 | 0.27 | 0.29 |
| IVF-PQ 48 B (1788 lists) + exact re-score | nprobe=16, k_factor=16 | 13.7 | 633 | 3162 | 0.705 | 0.33 | 0.42 |

