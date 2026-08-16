# Playground S6E8 — Predicting Smartphone Addiction

Kaggle: `playground-series-s6e8` · Swag · **deadline 2026-08-31**.
Metric: **ROC-AUC**. Binary target `addicted_label` (base rate 70.94% positive).

Private repo until the competition ends.

## Data

- `train.csv` 691,369 rows · `test.csv` 296,302 rows · `id` + 12 features + target.
- Numeric (with missing values): age, daily_screen_time_hours, social_media_hours,
  gaming_hours, work_study_hours, sleep_hours, notifications_per_day, app_opens_per_day,
  weekend_screen_time.
- Categorical: gender, stress_level, academic_work_impact.
- Sample submission = the constant base rate (0.7094) — i.e. AUC 0.5 baseline.

## Baseline (`train.py`)

LightGBM, 5-fold stratified CV, native NaN + categorical handling, lr=0.02, 63 leaves,
early stopping (150), predictions averaged across folds.

| fold | AUC | best_iter |
|---|---|---|
| 0 | 0.96319 | 4963 |
| 1 | 0.96403 | 4948 |
| 2 | 0.96433 | 4667 |
| 3 | 0.96484 | 4028 |
| 4 | 0.96377 | 4380 |

**OOF AUC = 0.96403** (~9.5 min). `submission.csv` written (296,302 rows).

## Next steps

- Feature engineering: ratios (screen_time/sleep, social/total, notifications/app_opens),
  per-hour interactions, missing-value indicator flags, age buckets.
- Try CatBoost + XGBoost and blend; tune LightGBM (num_leaves, min_child_samples).
- Several folds hit near the 5000-tree cap → raise n_estimators or lr and re-check.
- Not yet submitted to the leaderboard (submitting is an outward action — needs the user
  to run `kaggle competitions submit` or confirm).

## Environment

- `.venv` with pandas, numpy, scikit-learn, lightgbm. Data + venv gitignored.
- Reproduce: `python train.py`.
