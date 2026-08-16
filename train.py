"""Playground S6E8 - Predicting Smartphone Addiction. Metric: ROC-AUC.

LightGBM baseline: 5-fold stratified CV, native NaN + categorical handling,
out-of-fold AUC reported, test predictions averaged across folds.

Usage: python train.py
"""

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).parent
DATA = ROOT / "data"
TARGET = "addicted_label"
CAT_COLS = ["gender", "stress_level", "academic_work_impact"]
N_FOLDS = 5
SEED = 42

PARAMS = dict(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    learning_rate=0.02,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    n_estimators=5000,
    n_jobs=-1,
    verbose=-1,
    seed=SEED,
)


def load():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        # align test categories to train's so codes match
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)
    return train, test


def main():
    t0 = time.time()
    train, test = load()
    features = [c for c in train.columns if c not in ("id", TARGET)]
    X, y = train[features], train[TARGET].values
    X_test = test[features]

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(
            X.iloc[tr], y[tr],
            eval_set=[(X.iloc[va], y[va])],
            eval_metric="auc",
            categorical_feature=CAT_COLS,
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS
        auc = roc_auc_score(y[va], oof[va])
        print(f"fold {fold}: AUC={auc:.5f}  best_iter={model.best_iteration_}")

    cv_auc = roc_auc_score(y, oof)
    print(f"\nOOF AUC = {cv_auc:.5f}   ({time.time()-t0:.0f}s)")

    sub = pd.DataFrame({"id": test["id"], TARGET: test_pred})
    out = ROOT / "submission.csv"
    sub.to_csv(out, index=False)
    print(f"wrote {out}  ({len(sub)} rows)")


if __name__ == "__main__":
    main()
