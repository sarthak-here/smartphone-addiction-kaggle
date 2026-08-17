"""S6E8 - Optuna hyperparameter search for LightGBM (metric: ROC-AUC).

Resumable: the study is persisted to a SQLite DB so the search can run in
time-bounded chunks (the runner caps single commands at ~10 min) and accumulate
to the target trial count.

  python tune.py search [max_seconds]   # run trials until target or time budget
  python tune.py final                  # retrain best params 5-fold -> submission
  python tune.py status                 # show trial count + best value

Search runs on a stratified subsample (fast, preserves AUC ranking); the final
retrain uses the full training set with 5-fold CV.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna_integration.lightgbm import LightGBMPruningCallback
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
TARGET_TRIALS = 100
SEARCH_SUBSAMPLE = 120_000  # rows used for the search holdout (speed)

STORAGE = f"sqlite:///{(ROOT / 'optuna_s6e8.db').as_posix()}"
STUDY_NAME = "s6e8_lgbm_v2"


def load():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)
    return train, test


def get_study():
    return optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30),
        load_if_exists=True,
    )


def n_complete(study):
    return sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def cmd_search(max_seconds: float):
    train, _ = load()
    features = [c for c in train.columns if c not in ("id", TARGET)]
    X, y = train[features], train[TARGET].values

    # Stratified subsample for a fast, consistent search objective.
    if len(X) > SEARCH_SUBSAMPLE:
        X, _, y, _ = train_test_split(
            X, y, train_size=SEARCH_SUBSAMPLE, stratify=y, random_state=SEED
        )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            objective="binary", metric="auc", boosting_type="gbdt",
            n_jobs=-1, verbose=-1, seed=SEED, n_estimators=1000, max_bin=255,
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 16, 160),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 300),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            subsample_freq=trial.suggest_int("subsample_freq", 0, 7),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
        )
        model = lgb.LGBMClassifier(**params)
        pruning = LightGBMPruningCallback(trial, "auc", valid_name="valid_0")
        model.fit(
            X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
            categorical_feature=CAT_COLS,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0), pruning],
        )
        return roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])

    study = get_study()
    done = n_complete(study)
    remaining = max(0, TARGET_TRIALS - done)
    if remaining == 0:
        print(f"Target {TARGET_TRIALS} trials already reached (best={study.best_value:.5f}).")
        return
    t0 = time.time()
    print(f"Search: {done}/{TARGET_TRIALS} complete; running up to {remaining} more "
          f"or {max_seconds:.0f}s.")
    study.optimize(objective, n_trials=remaining, timeout=max_seconds,
                   show_progress_bar=False)
    done = n_complete(study)
    print(f"Now {done}/{TARGET_TRIALS} complete. best={study.best_value:.5f} "
          f"({time.time()-t0:.0f}s this chunk)")


def cmd_final():
    train, test = load()
    features = [c for c in train.columns if c not in ("id", TARGET)]
    X, y = train[features], train[TARGET].values
    X_test = test[features]

    study = get_study()
    best = dict(study.best_params)
    print(f"Best holdout AUC={study.best_value:.5f} over {n_complete(study)} trials")
    (ROOT / "best_params.json").write_text(json.dumps(study.best_params, indent=2))

    best.update(objective="binary", metric="auc", boosting_type="gbdt",
                n_jobs=-1, verbose=-1, seed=SEED, n_estimators=8000)
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    t0 = time.time()
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        model = lgb.LGBMClassifier(**best)
        model.fit(
            X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], eval_metric="auc",
            categorical_feature=CAT_COLS,
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"fold {fold}: AUC={roc_auc_score(y[va], oof[va]):.5f} "
              f"best_iter={model.best_iteration_}")
    cv = roc_auc_score(y, oof)
    print(f"\nFinal OOF AUC = {cv:.5f}  ({time.time()-t0:.0f}s)")
    pd.DataFrame({"id": test["id"], TARGET: test_pred}).to_csv(
        ROOT / "submission_optuna.csv", index=False)
    print("wrote submission_optuna.csv + best_params.json")


def cmd_status():
    study = get_study()
    print(f"{n_complete(study)}/{TARGET_TRIALS} complete trials; "
          f"best={study.best_value:.5f}" if study.trials else "no trials yet")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "search"
    if cmd == "search":
        cmd_search(float(sys.argv[2]) if len(sys.argv) > 2 else 480.0)
    elif cmd == "final":
        cmd_final()
    else:
        cmd_status()
