# Predicting Smartphone Addiction

An end-to-end tabular machine-learning solution for Kaggle Playground Series S6E8. The project predicts `addicted_label` from demographic, behavioral, screen-time, sleep, notification, and app-usage features using leakage-safe cross-validation and model stacking.

## Best result

| Model | OOF AUC | Public LB | Private LB |
|---|---:|---:|---:|
| LightGBM baseline | 0.96403 | 0.96551 | 0.96530 |
| Optuna LightGBM | 0.96412 | 0.96589 | 0.96571 |
| LightGBM + XGBoost blend | 0.96486 | 0.96627 | 0.96604 |
| **Five-model logistic meta-stack** | **0.96516** | **0.96661** | **0.96634** |

The final stack improved consistently from local out-of-fold validation to both public and private leaderboard evaluation.

## Approach

- Five-fold stratified cross-validation with a fixed seed.
- Native categorical handling for LightGBM.
- Engineered behavioral ratios and interaction features.
- Optuna search for LightGBM hyperparameters.
- LightGBM, XGBoost, CatBoost, and neural-network base learners.
- Leakage-safe out-of-fold target encoding for categorical interactions.
- Logistic-regression stacking over aligned base-model OOF predictions.
- Test predictions averaged across folds.

Target encoding is learned only from each fold's training partition. The meta-model is also trained out of fold, keeping the reported `0.96516` meta AUC honest.

## Data

The competition data contains:

- 691,369 training rows
- 296,302 test rows
- 12 input features plus `id`
- binary target: `addicted_label`
- numerical features with missing values
- categorical features: `gender`, `stress_level`, and `academic_work_impact`

Download the competition files from Kaggle and place them here:

```text
data/
  train.csv
  test.csv
  sample_submission.csv
```

Competition data and generated predictions are intentionally excluded from Git.

## Repository structure

```text
train.py          Five-fold LightGBM baseline
tune.py           Resumable Optuna tuning and final retraining
stack.py          Feature engineering, base models, and rank blend
stack2.py         OOF target encoding and logistic meta-stacker
mlp.py            GPU tabular neural-network experiment
best_params.json  Best saved LightGBM parameters
STATUS.md         Dataset notes and baseline experiment record
```

## Reproduce

Install the main dependencies:

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna torch
```

Run the baseline:

```bash
python train.py
```

Train the base models and build the first blend:

```bash
python stack.py lgb
python stack.py xgb
python stack.py cat
python mlp.py
python stack.py blend
```

Train the target-encoded model and final stack:

```bash
python stack2.py lgbte
python stack2.py stack
```

The final command writes `submission_stack.csv`.

## Competition

- Competition: Playground Series S6E8
- Metric: ROC-AUC
- Kaggle slug: `playground-series-s6e8`
