"""S6E8 - feature engineering + multi-model blend (metric: ROC-AUC).

Shared FE, then one model per invocation (so each fits the runner's time budget),
saving out-of-fold + test predictions to .npy. A final `blend` step combines them.

  python stack.py lgb       # LightGBM (Optuna best_params.json) on engineered features
  python stack.py xgb       # XGBoost (native categorical)
  python stack.py cat       # CatBoost
  python stack.py blend     # combine saved OOF/test preds -> submission_blend.csv

Every model uses the same folds (StratifiedKFold seed=42) so OOF preds align for blending.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).parent
DATA = ROOT / "data"
PRED = ROOT / "preds"
PRED.mkdir(exist_ok=True)
TARGET = "addicted_label"
BASE_CATS = ["gender", "stress_level", "academic_work_impact"]
SEED = 42
N_FOLDS = 5
EPS = 1e-3


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    scr = d["daily_screen_time_hours"]
    soc, gam, wrk = d["social_media_hours"], d["gaming_hours"], d["work_study_hours"]
    slp, notif, opens = d["sleep_hours"], d["notifications_per_day"], d["app_opens_per_day"]
    wknd = d["weekend_screen_time"]

    d["leisure_hours"] = soc + gam
    d["activity_sum"] = soc + gam + wrk
    d["screen_minus_components"] = scr - (soc + gam + wrk)
    d["social_ratio"] = soc / (scr + EPS)
    d["gaming_ratio"] = gam / (scr + EPS)
    d["work_ratio"] = wrk / (scr + EPS)
    d["leisure_ratio"] = (soc + gam) / (scr + EPS)
    d["screen_sleep_ratio"] = scr / (slp + EPS)
    d["weekend_ratio"] = wknd / (scr + EPS)
    d["weekend_delta"] = wknd - scr
    d["notif_per_open"] = notif / (opens + EPS)
    d["notif_per_screen"] = notif / (scr + EPS)
    d["opens_per_screen"] = opens / (scr + EPS)
    d["awake_hours"] = 24.0 - slp
    d["screen_per_awake"] = scr / (24.0 - slp + EPS)
    d["accounted_hours"] = soc + gam + wrk + slp
    d["free_hours"] = 24.0 - (soc + gam + wrk + slp)
    d["notif_per_awake"] = notif / (24.0 - slp + EPS)
    # missing-value flags
    for c in ["daily_screen_time_hours", "social_media_hours", "gaming_hours",
              "work_study_hours", "sleep_hours", "notifications_per_day",
              "app_opens_per_day", "weekend_screen_time", "age"]:
        d[f"{c}_isna"] = df[c].isna().astype("int8")
    return d


def load():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    train = engineer(train)
    test = engineer(test)
    features = [c for c in train.columns if c not in ("id", TARGET)]
    return train, test, features


def folds(y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))


def save(name, oof, test_pred, auc):
    np.save(PRED / f"oof_{name}.npy", oof)
    np.save(PRED / f"test_{name}.npy", test_pred)
    (PRED / f"auc_{name}.txt").write_text(f"{auc:.6f}")
    print(f"saved {name}: OOF AUC = {auc:.5f}")


def run_lgb():
    import lightgbm as lgb
    train, test, features = load()
    for c in BASE_CATS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)
    X, y, Xt = train[features], train[TARGET].values, test[features]
    bp = json.loads((ROOT / "best_params.json").read_text())
    bp.update(objective="binary", metric="auc", n_jobs=-1, verbose=-1, seed=SEED,
              n_estimators=8000)
    oof, tp = np.zeros(len(train)), np.zeros(len(test))
    for f, (tr, va) in enumerate(folds(y)):
        m = lgb.LGBMClassifier(**bp)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], eval_metric="auc",
              categorical_feature=BASE_CATS,
              callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        tp += m.predict_proba(Xt)[:, 1] / N_FOLDS
        print(f"  fold {f}: {roc_auc_score(y[va], oof[va]):.5f}")
    save("lgb", oof, tp, roc_auc_score(y, oof))


def run_xgb():
    import xgboost as xgb
    train, test, features = load()
    for c in BASE_CATS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)
    X, y, Xt = train[features], train[TARGET].values, test[features]
    params = dict(n_estimators=6000, learning_rate=0.03, max_depth=8,
                  subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
                  reg_alpha=1.0, reg_lambda=3.0, tree_method="hist",
                  enable_categorical=True, eval_metric="auc", n_jobs=-1,
                  early_stopping_rounds=150, random_state=SEED)
    oof, tp = np.zeros(len(train)), np.zeros(len(test))
    for f, (tr, va) in enumerate(folds(y)):
        m = xgb.XGBClassifier(**params)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        tp += m.predict_proba(Xt)[:, 1] / N_FOLDS
        print(f"  fold {f}: {roc_auc_score(y[va], oof[va]):.5f}")
    save("xgb", oof, tp, roc_auc_score(y, oof))


def run_cat():
    import gc
    from catboost import CatBoostClassifier, Pool
    train, test, features = load()
    for c in BASE_CATS:
        train[c] = train[c].astype(str).fillna("NA")
        test[c] = test[c].astype(str).fillna("NA")
    # float32 for numeric columns to roughly halve memory (avoid OOM on this size).
    num_cols = [c for c in features if c not in BASE_CATS]
    for c in num_cols:
        train[c] = train[c].astype("float32")
        test[c] = test[c].astype("float32")
    X, y, Xt = train[features], train[TARGET].values, test[features]
    cat_idx = [features.index(c) for c in BASE_CATS]
    Xt_pool = Pool(Xt, cat_features=cat_idx)
    oof, tp = np.zeros(len(train)), np.zeros(len(test))
    for f, (tr, va) in enumerate(folds(y)):
        m = CatBoostClassifier(iterations=1500, learning_rate=0.07, depth=6,
                               l2_leaf_reg=6.0, loss_function="Logloss",
                               eval_metric="AUC", random_seed=SEED,
                               od_type="Iter", od_wait=100, verbose=0,
                               boosting_type="Plain", max_ctr_complexity=1,
                               thread_count=4, used_ram_limit="6gb",
                               allow_writing_files=False)
        m.fit(Pool(X.iloc[tr], y[tr], cat_features=cat_idx),
              eval_set=Pool(X.iloc[va], y[va], cat_features=cat_idx))
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        tp += m.predict_proba(Xt_pool)[:, 1] / N_FOLDS
        print(f"  fold {f}: {roc_auc_score(y[va], oof[va]):.5f}", flush=True)
        del m
        gc.collect()
    save("cat", oof, tp, roc_auc_score(y, oof))


def run_blend():
    from scipy.optimize import minimize
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].values
    names = [n for n in ("lgb", "xgb", "cat") if (PRED / f"oof_{n}.npy").exists()]
    if not names:
        print("no model preds found; run lgb/xgb/cat first")
        return
    oofs = {n: np.load(PRED / f"oof_{n}.npy") for n in names}
    tests = {n: np.load(PRED / f"test_{n}.npy") for n in names}
    for n in names:
        print(f"  {n}: OOF AUC {roc_auc_score(y, oofs[n]):.5f}")

    # rank-average is scale-free and robust for AUC
    def rank(a):
        return pd.Series(a).rank(pct=True).values
    O = np.vstack([rank(oofs[n]) for n in names])
    T = np.vstack([rank(tests[n]) for n in names])

    def neg_auc(w):
        w = np.clip(w, 0, None)
        if w.sum() == 0:
            return 0.0
        return -roc_auc_score(y, (w[:, None] * O).sum(0) / w.sum())

    res = minimize(neg_auc, np.ones(len(names)) / len(names), method="Nelder-Mead")
    w = np.clip(res.x, 0, None)
    w = w / w.sum()
    blend_oof = (w[:, None] * O).sum(0)
    blend_test = (w[:, None] * T).sum(0)
    print("  weights:", {n: round(float(wi), 3) for n, wi in zip(names, w)})
    print(f"\nBlended OOF AUC = {roc_auc_score(y, blend_oof):.5f}")
    pd.DataFrame({"id": test["id"], TARGET: blend_test}).to_csv(
        ROOT / "submission_blend.csv", index=False)
    print("wrote submission_blend.csv")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "blend"
    t0 = time.time()
    {"lgb": run_lgb, "xgb": run_xgb, "cat": run_cat, "blend": run_blend}[cmd]()
    print(f"[{cmd} done in {time.time()-t0:.0f}s]")
