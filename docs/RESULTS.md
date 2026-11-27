# Development-run results (synthetic only)

These are actual results from one development run, not evidence of real-world fraud detection. Test outputs were inspected during development; they are not an independent final benchmark. Regenerate on your machine; versions may change results.

Environment: Python 3.13.16, NumPy 2.5.5, scikit-learn 1.9.1, FastAPI 0.165.0; seed 1729; 9,000 synthetic records.

- Train: 5,400 rows, 108 positive labels; 2025-01-01T00:00:00+00:00 through 2025-03-16T23:40:00+00:00.
- Validation: 1,800 rows, 34 positive labels; 2025-03-17T00:00:00+00:00 through 2025-04-10T23:40:00+00:00.
- Test: 1,800 rows, 45 positive labels; 2025-04-11T00:00:00+00:00 through 2025-05-05T23:40:00+00:00.

Model choice used validation average precision only:

| Model | Validation AP | Validation Brier |
| --- | ---: | ---: |
| Logistic regression | 0.0287 | 0.0187 |
| Random forest | 0.0231 | 0.0189 |

Selected: **Logistic regression**. Validation cutoff: 0.043810, targeting 5% review. Validation actual review: 90/1800 (5.0%).

At that unchanged cutoff, synthetic test review: 167/1800 (9.3%); precision 4.8%; recall 17.8%; TP 8, FP 159, FN 37, TN 1596; test AP 0.0625, Brier 0.0242.

The test workload differs from the validation target. This is a small, deliberately invented example; the label mechanism, simulated distribution changes, limited positives and dev-time inspection prevent generalization to real data. Do not report these as real fraud outcomes.
