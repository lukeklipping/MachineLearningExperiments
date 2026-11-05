"""
Test DLinear on a univariate time series, optionally with lossy compression.

Examples:
  python dlinear_test.py                          # downloads ETTh1, column OT
  python dlinear_test.py --csv my.csv --column value
  python dlinear_test.py --quant-bits 4           # compress inputs to 16 levels
  python dlinear_test.py --sweep                  # compare several bit depths

Compression is applied to the INPUT series (what the model sees), while
forecasts are always scored against the ORIGINAL, uncompressed values.
"""
import argparse
import os
import urllib.request

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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


def quantize(x, bits):
    """Lossy compression: uniform quantization to 2**bits levels.
    Returns reconstructed series and the compression ratio vs 64-bit floats."""
    lo, hi = x.min(), x.max()
    levels = 2 ** bits - 1
    codes = np.round((x - lo) / (hi - lo) * levels)
    recon = codes / levels * (hi - lo) + lo
    return recon, 64.0 / bits


def make_windows(inputs, targets, L, T):
    """Sliding windows: X from `inputs`, Y from `targets`."""
    n = len(inputs) - L - T + 1
    X = np.stack([inputs[i:i + L] for i in range(n)])
    Y = np.stack([targets[i + L:i + L + T] for i in range(n)])
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32))


# ---------------------------------------------------------------- metrics
def mase(y_true, y_pred, y_train_naive_scale):
    return np.mean(np.abs(y_true - y_pred)) / y_train_naive_scale


# ---------------------------------------------------------------- run
def run(raw, bits, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n = len(raw)
    n_train, n_val = int(n * 0.7), int(n * 0.1)

    # Standardize using TRAIN statistics only (avoids leakage).
    mean, std = raw[:n_train].mean(), raw[:n_train].std()
    raw_s = (raw - mean) / std

    if bits is None:
        inp_s, ratio = raw_s, 1.0
    else:
        inp_s, ratio = quantize(raw_s, bits)

    L, T = args.input_len, args.horizon
    sl = lambda a, b: slice(a, b)
    # Windows overlap split borders by L so each split has full context.
    X_tr, Y_tr = make_windows(inp_s[sl(0, n_train)], raw_s[sl(0, n_train)], L, T)
    X_va, Y_va = make_windows(inp_s[sl(n_train - L, n_train + n_val)],
                              raw_s[sl(n_train - L, n_train + n_val)], L, T)
    X_te, Y_te = make_windows(inp_s[sl(n_train + n_val - L, n)],
                              raw_s[sl(n_train + n_val - L, n)], L, T)

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

    # Naive baseline: repeat last observed (compressed) value.
    naive = np.repeat(X_te[:, -1:].numpy(), T, axis=1)
    # MASE scale: mean abs one-step change of the raw training series.
    scale = np.mean(np.abs(np.diff(raw_s[:n_train])))

    return {
        "bits": "none" if bits is None else bits,
        "ratio": round(ratio, 1),
        "epochs": epoch + 1,
        "MSE": float(np.mean((true - pred) ** 2)),
        "MAE": float(np.mean(np.abs(true - pred))),
        "MASE": float(mase(true, pred, scale)),
        "naive_MAE": float(np.mean(np.abs(true - naive))),
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
    p.add_argument("--quant-bits", type=int, default=None)
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    raw = load_series(args.csv, args.column)
    print(f"Series length: {len(raw)}")

    settings = [None, 8, 6, 4, 3, 2] if args.sweep else [args.quant_bits]
    rows = [run(raw, b, args) for b in settings]
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nMetrics are in standardized units, scored vs ORIGINAL data. "
          "MASE < 1 beats the naive one-step benchmark.")


if __name__ == "__main__":
    main()
