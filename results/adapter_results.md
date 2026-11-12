# Part 3 query adapter

Train 430 patents (2024-02-15 to 2024-12-05), validation 80 (2024-12-12 to 2025-01-30), test 130 (2025-02-06 to 2025-05-01). Each abstract ranks all claims in the index; its own patent's claims are the right answers. Best epoch 30 of 300; ||W - I|| = 0.647, ||b|| = 0.069.

| Test metric | Before (identity) | After | Change |
|---|---|---|---|
| Own claim ranked 1st | 0.9077 | 0.9000 | -0.0077 |
| Own claim in top 10 | 0.9385 | 0.9462 | +0.0077 |
| MRR@10 | 0.9205 | 0.9150 | -0.0055 |
| nDCG@10 | 0.8541 | 0.8544 | +0.0003 |
| Mean similarity, own claims | 0.8962 | 0.6982 | -0.1980 |
| Mean similarity, other patents' claims | 0.6856 | 0.3653 | -0.3203 |
| Gap (own - other) | 0.2106 | 0.3329 | +0.1223 |
