# Freight Rate Prediction — Spotter Labs ML Engineer Assessment
Author: Ayesha Shahid

Original assessment instructions: see `freight-rate-ml-assessment.pdf`.
This README documents my actual approach and how to run the solution.

## How to run

```bash
python -m pip install -r requirements.txt
python train_predict.py
```

This trains the model, writes `validation_predictions.csv`, and fills
`predicted_rate` into `data/december-chart-inputs.csv`.

Then validate the outputs with Spotter's scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv
```

This generates `scorer_results/candidate_december.png`.

## Approach

**Validation strategy — time-based holdout, not random 80/20:**
`train-test.csv` spans January–October 2025 and the real task (`validation.csv`)
covers November–December 2025 — the future relative to training data. A random
split would let later rows sit in "training" while earlier rows get held out,
testing the model on data older than what it trained on, which leaks
information and overstates accuracy. Instead, the last 8 weeks of
`train-test.csv` (2025-09-06 to 2025-10-31) are held out as a stand-in
"future," and the model is validated by predicting that window from
everything before it.

**Model:** XGBoost regressor (500 trees, depth 5, learning rate 0.05).
Distance, weight, equipment, pickup/delivery, and calendar features
(month, day-of-week, day-of-year) are tabular with a strongly non-linear,
interaction-heavy relationship to rate — XGBoost handles this natively
without feature scaling and is the standard choice for pricing/rate
prediction on structured data at this scale (~48K rows).

**Feature selection — dropped `market_index` and `quote_signal`:**
Both features showed near-zero raw correlation with `posted_rate` (~0.03–0.04).
An ablation confirmed removing them *improves* holdout accuracy:

| Feature set | MAE | MAPE |
|---|---|---|
| Full (incl. market_index, quote_signal) | $155.37 | 8.71% |
| Structural only (dropped) | **$147.90** | **6.90%** |

Near-zero-signal features let the model fit noise during training that
doesn't generalize — removing them reduces overfitting. This also means
one model architecture serves both `validation.csv` (which has those
columns) and `december-chart-inputs.csv` (which never did).

**Data quality fixes:**
- 292 rows had negative `weight` (impossible for freight weight — a
  sign-flip entry error) → corrected with absolute value rather than
  dropped, preserving the rest of each row's valid data.
- Missing `weight` values (300 in train, 165 in validation) imputed with
  the training median (robust to the right-skew typical of freight weight).
- 183 rows with `posted_rate` >3 std from the mean were flagged but kept —
  unlike negative weight, an unusually high rate isn't necessarily an
  error (could be a real oversized/rush load), and XGBoost's tree splits
  are naturally more robust to outliers than a linear model would be.

**Feature importance (final model):** `distance` dominates at ~79–80%,
consistent with its 0.91 raw correlation with `posted_rate`. Equipment
type and lane (pickup/delivery) are secondary drivers.

## Holdout validation metrics
MAE ~$153, RMSE ~$666, MAPE ~7.1% (holdout: 2025-09-06 to 2025-10-31).

Full metrics and feature importances are saved to `metrics.json` on each run.
