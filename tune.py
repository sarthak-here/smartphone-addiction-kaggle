"""S6E8 - Optuna hyperparameter search for LightGBM (metric: ROC-AUC).

Search phase: 100 trials on a fixed stratified 80/20 holdout (fast, with early
stopping + median pruning). Final phase: retrain the best params with full 5-fold
CV, average test predictions, and write submission.csv.

Usage: python tune.py [n_trials]
"""

import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).parent
DATA = ROOT / "data"
TARGET = "addicted_label"
CAT_COLS = ["gender", "stress_level", "academic_work_impact"]
SEED = 42
N_FOLDS = 5


def load():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)
    return train, test


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    t0 = time.time()
    train, test = load()
    features = [c for c in train.columns if c not in ("id", TARGET)]
    X, y = train[features], train[TARGET].values
    X_test = test[features]

    # Fixed holdout for the search (fast + consistent objective across trials).
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            objective="binary",
            metric="auc",
            boosting_type="gbdt",
            n_jobs=-1,
            verbose=-1,
            seed=SEED,
            n_estimators=4000,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 16, 255),
            max_depth=trial.suggest_int("max_depth", 3, 14),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 400),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            subsample_freq=trial.suggest_int("subsample_freq", 0, 7),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
            max_bin=trial.suggest_int("max_bin", 128, 511),
        )
        model = lgb.LGBMClassifier(**params)
        pruning = optuna.integration.LightGBMPruningCallback(trial, "auc", valid_name="valid_0")
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            categorical_feature=CAT_COLS,
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0), pruning],
        )
        pred = model.predict_proba(X_va)[:, 1]
        return roc_auc_score(y_va, pred)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=200),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"\nBest holdout AUC = {study.best_value:.5f}  ({len(study.trials)} trials, "
          f"{time.time()-t0:.0f}s)")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # --- Final: retrain best params with full 5-fold CV ---
    best = dict(study.best_params)
    best.update(objective="binary", metric="auc", boosting_type="gbdt",
                n_jobs=-1, verbose=-1, seed=SEED, n_estimators=6000)

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        model = lgb.LGBMClassifier(**best)
        model.fit(
            X.iloc[tr], y[tr],
            eval_set=[(X.iloc[va], y[va])],
            eval_metric="auc",
            categorical_feature=CAT_COLS,
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"fold {fold}: AUC={roc_auc_score(y[va], oof[va]):.5f}  "
              f"best_iter={model.best_iteration_}")

    cv_auc = roc_auc_score(y, oof)
    print(f"\nFinal OOF AUC = {cv_auc:.5f}   (total {time.time()-t0:.0f}s)")

    sub = pd.DataFrame({"id": test["id"], TARGET: test_pred})
    out = ROOT / "submission_optuna.csv"
    sub.to_csv(out, index=False)
    print(f"wrote {out}  ({len(sub)} rows)")

    # persist best params for reproducibility
    import json
    (ROOT / "best_params.json").write_text(json.dumps(study.best_params, indent=2))
    print("wrote best_params.json")


if __name__ == "__main__":
    main()
