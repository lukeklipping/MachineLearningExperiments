"""
Test DLinear on a univariate time series with real lossy compression.

Needs compressors.py in the same folder.  pip install torch numpy pandas PyWavelets scipy

Examples:
  python dlinear_test.py                                  # ETTh1, no compression
  python dlinear_test.py --csv my.csv --column value
  python dlinear_test.py --method wavelet --ratio 8       # one compressed run
  python dlinear_test.py --method all --sweep             # every method x ratio

How compression is applied:
  Every input window (the history the model sees) is compressed and
  reconstructed on its own. Forecast targets are never compressed, and all
  scores compare forecasts against the ORIGINAL values.
"""
import argparse
import os
import sys
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compressors.compress import COMPRESSORS, compress_windows

ETTH1_URL = ("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/"
             "main/ETT-small/ETTh1.csv")


# ---------------------------------------------------------------- model
class MovingAvg(nn.Module):
    """Rolling mean over time, with edge padding so output length == input."""
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):                       # x: (batch, L)
        pad_front = (self.k - 1) // 2
        pad_back = self.k - 1 - pad_front
        front = x[:, :1].repeat(1, pad_front)
        back = x[:, -1:].repeat(1, pad_back)
        padded = torch.cat([front, x, back], dim=1)
        return padded.unfold(1, self.k, 1).mean(dim=2)


class DLinear(nn.Module):
    def __init__(self, input_len, horizon, kernel=25):
        super().__init__()
        self.avg = MovingAvg(kernel)
        self.trend_layer = nn.Linear(input_len, horizon)
        self.remainder_layer = nn.Linear(input_len, horizon)

    def forward(self, x):                       # x: (batch, L)
        trend = self.avg(x)
        remainder = x - trend
        return self.trend_layer(trend) + self.remainder_layer(remainder)


# ---------------------------------------------------------------- data
def load_series(csv_path, column):
    if csv_path is None:
        csv_path = "ETTh1.csv"
        if not os.path.exists(csv_path):
            print(f"Downloading ETTh1 to {csv_path} ...")
            urllib.request.urlretrieve(ETTH1_URL, csv_path)
        column = column or "OT"
    df = pd.read_csv(csv_path)
    if column is None:
        numeric = df.select_dtypes("number").columns
        column = numeric[0]
        print(f"No --column given, using first numeric column: {column}")
    return df[column].astype(float).to_numpy()


def make_windows(series, L, T):
    """Sliding windows over one series: inputs X (n, L), targets Y (n, T)."""
    n = len(series) - L - T + 1
    X = np.stack([series[i:i + L] for i in range(n)])
    Y = np.stack([series[i + L:i + L + T] for i in range(n)])
    return X, Y


def build_splits(raw, method, ratio, L, T):
    """Standardize, window, and compress the INPUT windows of each split.
    Returns ({split: (X, Y)}, effective compression ratio)."""
    n = len(raw)
    n_train, n_val = int(n * 0.7), int(n * 0.1)

    # Standardize using TRAIN statistics only (avoids leakage).
    raw_s = (raw - raw[:n_train].mean()) / raw[:n_train].std()

    # Windows overlap split borders by L so each split has full context.
    bounds = {"train": (0, n_train),
              "val": (n_train - L, n_train + n_val),
              "test": (n_train + n_val - L, n)}
    splits, eff_ratio = {}, 1.0
    for name, (a, b) in bounds.items():
        X, Y = make_windows(raw_s[a:b], L, T)
        if method is not None:
            X, eff_ratio = compress_windows(X, method, ratio)
        splits[name] = (X.astype(np.float32), Y.astype(np.float32))
    return splits, eff_ratio


# ---------------------------------------------------------------- run
def run(raw, method, ratio, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    L, T = args.input_len, args.horizon
    splits, eff_ratio = build_splits(raw, method, ratio, L, T)
    X_tr, Y_tr = (torch.from_numpy(a) for a in splits["train"])
    X_va, Y_va = (torch.from_numpy(a) for a in splits["val"])
    X_te, Y_te = (torch.from_numpy(a) for a in splits["test"])

    model = DLinear(L, T, args.kernel)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val, best_state, patience = float("inf"), None, 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X_tr[idx]), Y_tr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(X_va), Y_va).item()
        if val < best_val:
            best_val, patience = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                break
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred = model(X_te).numpy()
    true = Y_te.numpy()

    # Naive baseline: repeat the last value of the (compressed) history.
    naive = np.repeat(X_te[:, -1:].numpy(), T, axis=1)
    mae = float(np.mean(np.abs(true - pred)))
    naive_mae = float(np.mean(np.abs(true - naive)))

    return {
        "method": method or "none",
        "target": "-" if ratio is None else ratio,
        "ratio": round(eff_ratio, 1),
        "epochs": epoch + 1,
        "MSE": float(np.mean((true - pred) ** 2)),
        "MAE": mae,
        "naive_MAE": naive_mae,
        # Model error / naive error. Below 1 beats "repeat last value".
        "err_vs_naive": mae / naive_mae,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None)
    p.add_argument("--column", default=None)
    p.add_argument("--input-len", type=int, default=336)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--kernel", type=int, default=25)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="wavelet",
                   choices=list(COMPRESSORS) + ["all"])
    p.add_argument("--ratio", type=float, default=None,
                   help="target ratio: keep ~input_len/ratio values per window")
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    raw = load_series(args.csv, args.column)
    print(f"Series length: {len(raw)}")

    methods = list(COMPRESSORS) if args.method == "all" else [args.method]
    if args.sweep:
        jobs = [(None, None)] + [(m, r) for m in methods for r in (2, 4, 8, 16, 32)]
    elif args.ratio is None:
        jobs = [(None, None)]
    else:
        jobs = [(m, args.ratio) for m in methods]

    rows = []
    for m, r in jobs:
        print(f"running {m or 'none'} {r or ''} ...", flush=True)
        rows.append(run(raw, m, r, args))
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nratio = real storage ratio (includes coefficient positions); "
          "target = requested.\nScores are in standardized units against the "
          "ORIGINAL data; err_vs_naive < 1 beats repeat-last-value.")


if __name__ == "__main__":
    main()