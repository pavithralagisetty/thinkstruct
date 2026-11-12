# Part 3 query adapter (trained on summary queries)

Train 430 patents (2024-02-15 to 2024-12-05), validation 80 (2024-12-12 to 2025-01-30), test 130 (2025-02-06 to 2025-05-01). Each summary ranks all claims in the index; its own patent's claims are the right answers. Best epoch 10 of 300; ||W - I|| = 0.490, ||b|| = 0.026.

| Test metric | Before (identity) | After | Change |
|---|---|---|---|
| Own claim ranked 1st | 0.8846 | 0.8462 | -0.0385 |
| Own claim in top 10 | 0.9385 | 0.9385 | +0.0000 |
| MRR@10 | 0.8985 | 0.8754 | -0.0231 |
| nDCG@10 | 0.8124 | 0.8039 | -0.0084 |
| Mean similarity, own claims | 0.8510 | 0.7686 | -0.0824 |
| Mean similarity, other patents' claims | 0.6707 | 0.5268 | -0.1439 |
| Gap (own - other) | 0.1803 | 0.2418 | +0.0615 |
