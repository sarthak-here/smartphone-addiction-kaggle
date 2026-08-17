"""S6E8 - target encoding + logistic-regression stacking. Metric: ROC-AUC.

  python stack2.py lgbte   # LightGBM with OOF target-encoded cats+interactions
  python stack2.py stack   # LR meta-stacker over all base OOF/test preds

Target encoding is computed out-of-fold (per the shared StratifiedKFold seed=42)
with smoothing, so it does not leak. The LR stacker is fit with the same folds on
the base models' OOF predictions, so the reported meta AUC is honest.
"""

import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from stack import BASE_CATS, DATA, N_FOLDS, PRED, ROOT, SEED, TARGET, engineer, folds, save

SMOOTH = 20.0  # target-encoding smoothing (higher = more shrink to global mean)


def te_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Categorical + interaction string columns to target-encode."""
    d = pd.DataFrame(index=df.index)
    g, s, a = (df["gender"].astype("string").fillna("NA"),
               df["stress_level"].astype("string").fillna("NA"),
               df["academic_work_impact"].astype("string").fillna("NA"))
    d["te_gender"] = g
    d["te_stress"] = s
    d["te_academic"] = a
    d["te_gender_stress"] = g + "|" + s
    d["te_stress_academic"] = s + "|" + a
    d["te_gender_academic"] = g + "|" + a
    d["te_all3"] = g + "|" + s + "|" + a
    age_bin = pd.cut(df["age"], bins=10, labels=False).astype("string").fillna("NA")
    d["te_agebin_stress"] = age_bin + "|" + s
    return d


def oof_target_encode(tr_cat, y, te_cat, folds_list):
    """Return (oof_encoded_train, encoded_test) for one categorical column."""
    global_mean = y.mean()
    oof = np.full(len(tr_cat), global_mean, dtype=np.float64)
    for tr, va in folds_list:
        stats = pd.DataFrame({"c": tr_cat.iloc[tr].values, "y": y[tr]}).groupby("c")["y"]
        cnt, mean = stats.count(), stats.mean()
        enc = (cnt * mean + SMOOTH * global_mean) / (cnt + SMOOTH)
        oof[va] = tr_cat.iloc[va].map(enc).fillna(global_mean).values
    # test: encode with full-train stats
    stats = pd.DataFrame({"c": tr_cat.values, "y": y}).groupby("c")["y"]
    cnt, mean = stats.count(), stats.mean()
    enc = (cnt * mean + SMOOTH * global_mean) / (cnt + SMOOTH)
    test_enc = te_cat.map(enc).fillna(global_mean).values
    return oof, test_enc


def run_lgbte():
    import lightgbm as lgb
    import json
    train = engineer(pd.read_csv(DATA / "train.csv"))
    test = engineer(pd.read_csv(DATA / "test.csv"))
    y = train[TARGET].values
    fl = folds(y)

    te_tr, te_te = te_columns(train), te_columns(test)
    for col in te_tr.columns:
        oof_e, test_e = oof_target_encode(te_tr[col], y, te_te[col], fl)
        train[col] = oof_e
        test[col] = test_e

    for c in BASE_CATS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)
    features = [c for c in train.columns if c not in ("id", TARGET)]
    X, Xt = train[features], test[features]

    bp = json.loads((ROOT / "best_params.json").read_text())
    bp.update(objective="binary", metric="auc", n_jobs=-1, verbose=-1, seed=SEED,
              n_estimators=8000)
    oof, tp = np.zeros(len(train)), np.zeros(len(test))
    for f, (tr, va) in enumerate(fl):
        m = lgb.LGBMClassifier(**bp)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], eval_metric="auc",
              categorical_feature=BASE_CATS,
              callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        tp += m.predict_proba(Xt)[:, 1] / N_FOLDS
        print(f"  fold {f}: {roc_auc_score(y[va], oof[va]):.5f}", flush=True)
    save("lgbte", oof, tp, roc_auc_score(y, oof))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def run_stack():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].values
    names = [n for n in ("lgb", "xgb", "cat", "mlp", "lgbte")
             if (PRED / f"oof_{n}.npy").exists()]
    O = np.column_stack([_logit(np.load(PRED / f"oof_{n}.npy")) for n in names])
    T = np.column_stack([_logit(np.load(PRED / f"test_{n}.npy")) for n in names])
    for n in names:
        print(f"  {n}: OOF AUC {roc_auc_score(y, np.load(PRED / f'oof_{n}.npy')):.5f}")

    # LR meta-stacker, fit out-of-fold on the base OOF matrix (honest meta AUC).
    meta_oof = np.zeros(len(train))
    meta_test = np.zeros(len(test))
    for tr, va in folds(y):
        sc = StandardScaler().fit(O[tr])
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(sc.transform(O[tr]), y[tr])
        meta_oof[va] = lr.predict_proba(sc.transform(O[va]))[:, 1]
        meta_test += lr.predict_proba(sc.transform(T))[:, 1] / N_FOLDS
    print("  LR coefs (last fold):",
          {n: round(float(c), 3) for n, c in zip(names, lr.coef_[0])})
    print(f"\nStacked meta OOF AUC = {roc_auc_score(y, meta_oof):.5f}")
    pd.DataFrame({"id": test["id"], TARGET: meta_test}).to_csv(
        ROOT / "submission_stack.csv", index=False)
    print("wrote submission_stack.csv")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stack"
    t0 = time.time()
    {"lgbte": run_lgbte, "stack": run_stack}[cmd]()
    print(f"[{cmd} done in {time.time()-t0:.0f}s]")
