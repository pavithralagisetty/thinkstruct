# Part 3 results

## Query field: abstract

Train 510 patents (51 weeks, 2024-02-15 to 2025-01-30), test 130 patents (13 weeks, 2025-02-06 to 2025-05-01). Skipped: none; duplicates removed: 0. Training: 128 steps, batch 4, 373.4 s on mps:0.

| Metric | Base | Fine tuned | Change |
|---|---|---|---|
| Recall@1 | 0.8615 | 0.9385 | +0.0769 |
| Recall@10 | 0.9385 | 0.9769 | +0.0385 |
| MRR@10 | 0.8936 | 0.9522 | +0.0586 |
| nDCG@10 | 0.9050 | 0.9583 | +0.0533 |
| Triplet accuracy | 0.9615 | 0.9846 | +0.0231 |
| Mean similarity, correct pairs | 0.8844 | 0.8106 | -0.0737 |
| Mean similarity, wrong pairs | 0.7007 | 0.1205 | -0.5802 |
