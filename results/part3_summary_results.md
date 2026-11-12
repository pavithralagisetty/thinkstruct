# Part 3 fine tune on summaries

Questions: 640 summaries, one per patent (median 29 words, 95th percentile 34). Train 510 patents, 2514 rows (weeks 2024-02-15 to 2025-01-30); test 130 patents, 647 rows (2025-02-06 to 2025-05-01). Up to 5 claims per patent, main claim first. Training: 629 steps, batch 4, 1156.7 s on mps:0.

Wrong answers (each patent's first listed code (every patent here has exactly one)):

* train: main group 2462, subclass 37, any patent 15; candidates skipped: 5110 by the score check (623 rows had one), 1 with identical text
* test: main group 592, subclass 50, any patent 5; candidates skipped: 2600 by the score check (171 rows had one), 0 with identical text

Test: each test summary ranks every claim of the test patents; its own patent's claims are right. Both models read 512 tokens here.

| Metric | Base | Fine tuned | Change |
|---|---|---|---|
| Own claim ranked 1st | 0.9231 | 0.9231 | +0.0000 |
| Own claim in top 10 | 0.9615 | 0.9769 | +0.0154 |
| MRR@10 | 0.9354 | 0.9421 | +0.0067 |
| nDCG@10 | 0.8811 | 0.9153 | +0.0343 |
| Triplet accuracy | 1.0000 | 1.0000 | +0.0000 |
| Mean similarity, right claim | 0.8531 | 0.6994 | -0.1537 |
| Mean similarity, wrong claim | 0.7090 | 0.1279 | -0.5811 |

Note: Base triplet accuracy is 1.0 by construction: the score check keeps only wrong claims that base nomic scores below 90% of the right claim.
