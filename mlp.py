"""S6E8 - tabular MLP (categorical embeddings) for the blend. Metric: ROC-AUC.

Reuses the same feature engineering and folds as stack.py so its OOF/test preds
align with the GBDTs. Saves preds/oof_mlp.npy + preds/test_mlp.npy.

  python mlp.py
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from stack import BASE_CATS, DATA, N_FOLDS, PRED, SEED, TARGET, engineer, folds, save

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 8192
MAX_EPOCHS = 40
PATIENCE = 6


class TabMLP(nn.Module):
    def __init__(self, n_num, cardinalities):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(c, min(8, (c + 1) // 2 + 1)) for c in cardinalities
        ])
        emb_dim = sum(e.embedding_dim for e in self.embs)
        d = emb_dim + n_num
        self.body = nn.Sequential(
            nn.BatchNorm1d(d),
            nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x_num, x_cat):
        e = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        return self.body(torch.cat(e + [x_num], dim=1)).squeeze(1)


def main():
    t0 = time.time()
    train = engineer(pd.read_csv(DATA / "train.csv"))
    test = engineer(pd.read_csv(DATA / "test.csv"))
    features = [c for c in train.columns if c not in ("id", TARGET)]
    num_cols = [c for c in features if c not in BASE_CATS]
    y = train[TARGET].values.astype(np.float32)

    # Categorical -> integer codes (shared category set from train; unknown -> 0).
    cats, cardinalities = {}, []
    Xc_tr = np.zeros((len(train), len(BASE_CATS)), dtype=np.int64)
    Xc_te = np.zeros((len(test), len(BASE_CATS)), dtype=np.int64)
    for j, c in enumerate(BASE_CATS):
        vals = pd.Index(train[c].astype("string").fillna("NA").unique())
        mapping = {v: i + 1 for i, v in enumerate(vals)}  # 0 = unknown/NA
        cats[c] = mapping
        Xc_tr[:, j] = train[c].astype("string").fillna("NA").map(mapping).fillna(0).astype("int64")
        Xc_te[:, j] = test[c].astype("string").fillna("NA").map(mapping).fillna(0).astype("int64")
        cardinalities.append(len(mapping) + 1)

    # Numeric: replace inf, median-impute (global), standardize per fold below.
    Xn_tr = train[num_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    Xn_te = test[num_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    medians = Xn_tr.median()
    Xn_tr = Xn_tr.fillna(medians).values
    Xn_te = Xn_te.fillna(medians).values

    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    Xc_te_t = torch.tensor(Xc_te, device=DEVICE)

    for f, (tr, va) in enumerate(folds(y)):
        mu, sd = Xn_tr[tr].mean(0), Xn_tr[tr].std(0) + 1e-6
        def std(a):
            return np.clip((a - mu) / sd, -10, 10).astype(np.float32)
        xn_tr = torch.tensor(std(Xn_tr[tr]), device=DEVICE)
        xc_tr = torch.tensor(Xc_tr[tr], device=DEVICE)
        yt = torch.tensor(y[tr], device=DEVICE)
        xn_va = torch.tensor(std(Xn_tr[va]), device=DEVICE)
        xc_va = torch.tensor(Xc_tr[va], device=DEVICE)
        xn_te = torch.tensor(std(Xn_te), device=DEVICE)

        model = TabMLP(len(num_cols), cardinalities).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        lossf = nn.BCEWithLogitsLoss()
        n = len(tr)
        best_auc, best_state, wait = -1.0, None, 0

        for epoch in range(MAX_EPOCHS):
            model.train()
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, BATCH):
                idx = perm[i:i + BATCH]
                opt.zero_grad()
                out = model(xn_tr[idx], xc_tr[idx])
                loss = lossf(out, yt[idx])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                va_p = torch.sigmoid(model(xn_va, xc_va)).cpu().numpy()
            auc = roc_auc_score(y[va], va_p)
            if auc > best_auc + 1e-5:
                best_auc, best_state, wait = auc, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            oof[va] = torch.sigmoid(model(xn_va, xc_va)).cpu().numpy()
            test_pred += torch.sigmoid(model(xn_te, Xc_te_t)).cpu().numpy() / N_FOLDS
        print(f"  fold {f}: AUC={roc_auc_score(y[va], oof[va]):.5f} (best_epoch_auc={best_auc:.5f})", flush=True)

    save("mlp", oof, test_pred, roc_auc_score(y, oof))
    print(f"[mlp done in {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
