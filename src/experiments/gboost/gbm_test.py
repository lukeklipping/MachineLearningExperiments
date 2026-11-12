"""
Gradient boosting forecaster tested under lossy compression.

Put this file in src/gbm/ next to src/dlinear/ and src/compressors/.
Needs:  pip install scikit-learn   (plus what dlinear_test.py already needs)

It reuses build_splits() from dlinear_test.py, so it sees EXACTLY the same
train/val/test windows and the same compressed inputs as DLinear. That is
what makes the two models comparable.

Examples (run from src/):
  python gbm/gbm_test.py --csv ../test_data.csv --column traffic --method wavelet --ratio 8
  python gbm/gbm_test.py --csv ../test_data.csv --column traffic --method all --sweep

Strategy: "direct" forecasting. One boosted-tree model per future step
(horizon 96 -> 96 models). Each model gets features built from the compressed
history window:
  - the few most recent values
  - level/spread of the whole window and of the latest season
  - values from the SAME POINT IN THE CYCLE as the target time, one or more
    seasons earlier (e.g. same hour yesterday, two days ago, ...)
Set --season to your data's cycle length (24 for hourly data with a daily cycle).

Training is deterministic (no random subsampling), so repeated runs give
identical numbers, unlike DLinear. The 96 models are independent, so they are
trained in parallel (--jobs); results do not depend on the number of jobs.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from compressors.compress import COMPRESSORS
from experiments.dlinear.dlinear_test import build_splits, load_series


# ---------------------------------------------------------------- features
def make_features(X, h, season, n_recent=3):
    """Features for predicting step h (1-based) from windows X of shape (n, L)."""
    n, L = X.shape
    season = min(season, L)
    last = L - 1
    cols = [X[:, last - j] for j in range(min(n_recent, L))]
    recent = X[:, -season:]
    cols += [X.mean(axis=1), X.std(axis=1), recent.mean(axis=1), recent.std(axis=1),
             X[:, last] - recent.mean(axis=1)]
    # Target time is index last + h. Walk back whole seasons until the index
    # falls inside the window, then keep going back while we still can.
    k = int(np.ceil(h / season))
    pos = last + h - k * season
    while pos >= 0:
        cols.append(X[:, pos])
        pos -= season
    return np.column_stack(cols)


# ---------------------------------------------------------------- model
def fit_predict_one(F_tr, y_tr, F_va, y_va, F_te, args):
    """Fit one boosted model, pick the best number of trees on the validation
    set (like early stopping), and return test predictions at that point."""
    model = HistGradientBoostingRegressor(
        max_iter=args.iters, learning_rate=args.lr, max_leaf_nodes=args.leaves,
        min_samples_leaf=args.min_leaf, early_stopping=False, random_state=0)
    model.fit(F_tr, y_tr)

    best_i, best_err, best_pred = 0, np.inf, None
    for i, (p_va, p_te) in enumerate(zip(model.staged_predict(F_va),
                                         model.staged_predict(F_te))):
        err = np.mean(np.abs(p_va - y_va))
        if err < best_err:
            best_i, best_err, best_pred = i, err, p_te.copy()
    return best_pred, best_i + 1


def fit_step(h, X_tr, y_tr, X_va, y_va, X_te, args):
    """One forecast step = one independent model. Runs inside a worker."""
    # Tiny models train faster on one thread; parallelism comes from running
    # many steps at once instead.
    with threadpool_limits(limits=args.threads):
        return fit_predict_one(
            make_features(X_tr, h, args.season), y_tr,
            make_features(X_va, h, args.season), y_va,
            make_features(X_te, h, args.season), args)


def run(raw, method, ratio, args, naive_fixed):
    L, T = args.input_len, args.horizon
    t0 = time.time()
    splits, eff_ratio = build_splits(raw, method, ratio, L, T)
    X_tr, Y_tr = splits["train"]
    X_va, Y_va = splits["val"]
    X_te, Y_te = splits["test"]

    results = Parallel(n_jobs=args.jobs)(
        delayed(fit_step)(h, X_tr, Y_tr[:, h - 1], X_va, Y_va[:, h - 1], X_te, args)
        for h in range(1, T + 1))
    pred = np.column_stack([r[0] for r in results])
    trees = [r[1] for r in results]

    mae = float(np.mean(np.abs(Y_te - pred)))
    return {
        "method": method or "none",
        "target": "-" if ratio is None else ratio,
        "ratio": round(eff_ratio, 1),
        "avg_trees": round(float(np.mean(trees))),
        "MSE": float(np.mean((Y_te - pred) ** 2)),
        "MAE": mae,
        # Same fixed yardstick for every row: repeat-last-value on the
        # UNCOMPRESSED history. Unlike the old column, this is comparable.
        "err_vs_naive": mae / naive_fixed,
        "sec": round(time.time() - t0, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None)
    p.add_argument("--column", default=None)
    p.add_argument("--input-len", type=int, default=336)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--season", type=int, default=24,
                   help="cycle length in steps (24 = daily cycle of hourly data)")
    p.add_argument("--iters", type=int, default=200, help="max trees per model")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--leaves", type=int, default=15)
    p.add_argument("--min-leaf", type=int, default=20)
    p.add_argument("--jobs", type=int, default=-1,
                   help="models trained in parallel (-1 = all cores, 1 = serial)")
    p.add_argument("--threads", type=int, default=1,
                   help="threads used inside each model")
    p.add_argument("--method", default="wavelet", choices=list(COMPRESSORS) + ["all"])
    p.add_argument("--ratio", type=float, default=None)
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    raw = load_series(args.csv, args.column)
    print(f"Series length: {len(raw)}")

    # Fixed baseline from uncompressed data, shared by every row.
    base, _ = build_splits(raw, None, None, args.input_len, args.horizon)
    X_te, Y_te = base["test"]
    naive_fixed = float(np.mean(np.abs(Y_te - X_te[:, -1:])))

    methods = list(COMPRESSORS) if args.method == "all" else [args.method]
    jobs = [(None, None)]                      # control run, always first
    if args.sweep:
        jobs += [(m, r) for m in methods for r in (2, 4, 8, 16, 32)]
    elif args.ratio is not None:
        jobs += [(m, args.ratio) for m in methods]

    rows = []
    for m, r in jobs:
        print(f"running {m or 'none'} {r or ''} ...", flush=True)
        rows.append(run(raw, m, r, args, naive_fixed))
    df = pd.DataFrame(rows)
    df["vs_none"] = (df["MAE"] / df.loc[0, "MAE"]).round(3)
    print(df.to_string(index=False))
    print("\nvs_none = MAE divided by the uncompressed run's MAE (1.00 = compression "
          "cost nothing).\nerr_vs_naive uses one fixed baseline, so it is comparable "
          "across rows.\nScores are in standardized units against the ORIGINAL data.")


if __name__ == "__main__":
    main()