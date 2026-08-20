# Freight Rate Prediction — Spotter Labs ML Engineer Assessment
Author: Ayesha Shahid

Original assessment instructions: see `freight-rate-ml-assessment.pdf`.
This README documents my actual approach, findings, and how to reproduce
every result below.

## How to run

```bash
python -m pip install -r requirements.txt

python train_predict.py              # trains final model, writes
                                      # validation_predictions.csv and fills
                                      # predicted_rate into
                                      # data/december-chart-inputs.csv

python score.py --predictions validation_predictions.csv \
                 --december-predictions data/december-chart-inputs.csv
                                      # validates submission files, writes
                                      # scorer_results/candidate_december.png

python baseline.py                   # compares Linear Regression and
                                      # Random Forest against XGBoost

python error_analysis.py             # diagnoses why RMSE >> MAE

python walk_forward_validation.py    # multi-fold robustness check
```

## Repository contents

| File | Purpose |
|---|---|
| `train_predict.py` | Main pipeline: cleaning, feature engineering, final model, predictions |
| `baseline.py` | Model comparison — Linear Regression / Random Forest / XGBoost on the same holdout |
| `error_analysis.py` | Diagnoses the RMSE/MAE gap by identifying which loads drive it |
| `walk_forward_validation.py` | Multi-fold temporal validation for a more robust accuracy estimate |
| `score.py` | Spotter's official scorer (unmodified) |
| `metrics.json` | Holdout metrics + feature importances, regenerated on each `train_predict.py` run |

## Approach

### Validation strategy — time-based, not random 80/20
`train-test.csv` spans January–October 2025; the real task (`validation.csv`)
covers November–December 2025 — the future relative to training data. A
random split would let later rows sit in "training" while earlier rows are
held out, testing the model on data older than what it trained on. That
leaks information and overstates accuracy. Instead, the model is validated
by predicting a future window from everything before it — first with a
single 8-week holdout, then confirmed more rigorously with walk-forward
validation (below).

### Model choice — XGBoost, evaluated against baselines
`baseline.py` compares three models on the identical 56-day temporal
holdout:

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Linear Regression | $192.64 | $662.87 | 11.99% |
| Random Forest | $182.63 | $718.91 | 8.18% |
| **XGBoost (selected)** | **$152.90** | $666.36 | **7.05%** |

XGBoost wins clearly on MAE and MAPE — the metrics that reflect typical
prediction accuracy. Random Forest's higher RMSE despite a middling MAE
suggests it struggles more than XGBoost on the extreme-value loads
described below; XGBoost's sequential error-correction (each tree
correcting the previous ones' residuals) appears to handle those harder
cases better than Random Forest's simple averaging across independent
trees. Distance, weight, equipment, and lane have a non-linear,
interaction-heavy relationship with rate — exactly the setting where
gradient-boosted trees typically outperform linear models, and XGBoost is
the standard choice for pricing/rate prediction on structured tabular data
at this scale (~48K rows).

### Feature selection — dropped `market_index` and `quote_signal`
Both features showed near-zero raw correlation with `posted_rate`
(~0.03–0.04). An ablation confirmed removing them *improves* holdout
accuracy:

| Feature set | MAE | MAPE |
|---|---|---|
| Full (incl. market_index, quote_signal) | $155.37 | 8.71% |
| Structural only (dropped) | **$147.90** | **6.90%** |

Near-zero-signal features let the model fit noise during training that
doesn't generalize to new data — a mild form of overfitting. Removing them
improves generalization and, as a side benefit, means one model
architecture serves both `validation.csv` (which has those columns) and
`december-chart-inputs.csv` (which never did).

Note: `distance` dominates feature importance (~79-80%) and correlates
0.91 with `posted_rate`. That's importance and correlation, not a causal
claim — the model was never given a mechanism to test causality, only to
find predictive structure. Distance is very likely a genuine economic
driver of freight pricing (fuel, time, driver hours all scale with miles),
but the 79-80% figure specifically describes how much this model's
predictions rely on that feature, not a proven share of "true" cost.

### Data quality fixes
- 292 rows had negative `weight` (physically impossible for freight — a
  sign-flip entry error) → corrected with absolute value rather than
  dropped, preserving the rest of each row's valid data.
- Missing `weight` (300 rows train, 165 validation) imputed with the
  training median (robust to the right-skew typical of freight weight).
- 183 rows with `posted_rate` >3 std from the mean were flagged but kept
  — an unusually high rate isn't necessarily an error (could be a real
  oversized/rush/long-haul load), and unlike negative weight there's no
  domain rule that makes a high rate definitionally impossible.

### Why RMSE ($666) is much larger than MAE ($153)
`error_analysis.py` shows the top 1% of holdout loads (87 of 8,759) average
a **$5,435** prediction error, while the remaining 99% average just
**$100**. Because RMSE squares errors before averaging, these extreme
misses are amplified far more than MAE would show. Those 87 loads are
long-haul, cross-country lanes (avg. 1,457 mi) with posted rates averaging
$6,921 — 3-4x higher than what distance-based pricing predicts (e.g.
Baltimore→Oklahoma City, Montgomery→Fresno, Boston→Bakersfield). Mean
absolute error also increases steadily with distance bucket, from $49 on
sub-200-mile hauls to $293 on 2,000+ mile hauls — expected heteroscedasticity
in price data, since absolute error naturally scales with the magnitude of
the value being predicted.

**Interpretation:** MAE reflects the typical-case experience (~$153 off
per load); RMSE reflects the model's sensitivity to a small number of
high-value long-haul misses. A production version of this model would
likely benefit from a long-haul-specific pricing component or
distance-interaction features to close that gap.

### Robustness check — walk-forward validation
A single 8-week holdout can understate real-world variance. `walk_forward_validation.py`
runs three expanding-window folds, each validating on a full future month:

| Validation month | Train rows | MAE | RMSE | MAPE |
|---|---|---|---|---|
| August 2025 | 33,718 | $190.42 | $656.07 | 9.83% |
| September 2025 | 38,477 | $185.29 | $692.26 | 7.81% |
| October 2025 | 43,147 | $250.90 | $722.56 | 13.88% |
| **Mean ± Std** | | **$208.87 ± $36.49** | | **10.51% ± 3.09%** |

This is a more honest estimate of production performance than the single
holdout — the single-window MAE of ~$153 was somewhat optimistic. Notably,
October (the fold with the *most* training data) had the *worst* error,
worth further investigation with more time — possibly a seasonal effect or
a higher concentration of the extreme long-haul loads identified above.

## Business interpretation
An MAE of ~$150-210 per load, against a dataset where the median
`posted_rate` is ~$2,055, represents roughly 7-10% typical error (matching
the MAPE figures above). For most loads this is within a plausible
day-to-day pricing tolerance; the practical risk is concentrated in the
small subset of long-haul loads identified in the error analysis, where
misses can run into the thousands of dollars and would matter most for
margin on individual high-value shipments.

## Final holdout metrics (single 8-week window)
MAE $152.90, RMSE $666.36, MAPE 7.05% (holdout: 2025-09-06 to 2025-10-31).
See `metrics.json` for the exact values from the most recent run, and the
walk-forward table above for a more conservative multi-fold estimate.
