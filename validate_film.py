import argparse
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

from config import Config
from preprocessing import synthesize_lr, estimate_L
from data import _load_npy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", default=None)
    ap.add_argument("--num", type=int, default=600)
    ap.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args()
    cfg = Config()
    gt_dir = args.gt_dir or cfg.gt_dir

    gts = sorted(Path(gt_dir).glob("*.npy"))[: args.num]
    rng = np.random.default_rng(0)
    L_true, L_log, L_lin = [], [], []
    for p in gts:
        hr = _load_npy(p)
        L = float(rng.uniform(1.0, 12.0))            # spanning + exceeding provided range
        lr = synthesize_lr(hr, L, eps_clip=cfg.log_eps, rng=rng)
        L_true.append(L)
        L_log.append(estimate_L(lr, domain="log"))
        L_lin.append(estimate_L(lr, domain="linear"))

    r_log = spearmanr(L_true, L_log).statistic
    r_lin = spearmanr(L_true, L_lin).statistic
    print(f"N={len(L_true)}  Spearman  log-domain={r_log:.3f}  linear-domain={r_lin:.3f}")
    best = max(r_log, r_lin)
    if best < args.threshold:
        print(f"WARNING: best corr {best:.3f} < {args.threshold}. "
              f"FiLM branch is worthless -> run train.py --no_film for the deliverable model.")
        raise SystemExit(1)
    print(f"OK: L̂ tracks true L (best {best:.3f} >= {args.threshold}). FiLM conditioning justified.")


if __name__ == "__main__":
    main()
