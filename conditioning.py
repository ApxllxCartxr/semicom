"""
Does the conditioning signal survive the noise?

The regime features (grad_mean, var_med, var_iqr) were computed on GT. At
inference you only have the degraded input, and multiplicative speckle inflates
exactly those quantities. This script answers: for a given image, does the
feature computed on the noisy input tell you anything about the same feature on
the clean image -- or is it just reporting the noise level?

Usage:
    python conditioning_check.py --gt_dir train/train/GT --deg_dir train/train/LR

Reuses image_stats() from regime_analysis.py. Assumes GT and degraded files
share a basename; degraded may be lower resolution (features are
resolution-normalized enough for rank correlation).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from regime_analysis import image_stats


def load(p: Path) -> np.ndarray:
    if p.suffix.lower() == ".npy":
        a = np.load(p)
    else:
        from PIL import Image
        a = np.array(Image.open(p).convert("L"))
    a = np.squeeze(a)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    return a


def smoothed(a: np.ndarray) -> np.ndarray:
    """Cheap speckle suppressor: 2x2 box downsample. Averaging Gamma-distributed
    multiplicative noise reduces its variance by ~the number of looks, so
    features computed here should track signal rather than noise."""
    h, w = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
    return a[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--deg_dir", required=True)
    ap.add_argument("--limit", type=int, default=600)
    args = ap.parse_args()

    gt_paths = sorted(Path(args.gt_dir).rglob("*"))
    gt_paths = [p for p in gt_paths if p.suffix.lower() in
                {".npy", ".png", ".tif", ".tiff", ".jpg", ".bmp"}][: args.limit]

    rows = []
    for gp in gt_paths:
        matches = list(Path(args.deg_dir).rglob(gp.stem + ".*"))
        if not matches:
            continue
        g, d = load(gp), load(matches[0])
        sg, sd, sds = image_stats(g), image_stats(d), image_stats(smoothed(d))
        rows.append(
            {f"gt_{k}": v for k, v in sg.items()}
            | {f"deg_{k}": v for k, v in sd.items()}
            | {f"sm_{k}": v for k, v in sds.items()}
        )

    if not rows:
        raise SystemExit("No GT/degraded basename matches -- check --deg_dir")
    df = pd.DataFrame(rows)
    print(f"paired {len(df)} images\n")

    keys = list(image_stats(np.zeros((16, 16))).keys())
    print(f"{'feature':<12} {'raw deg vs GT':>14} {'smoothed vs GT':>16}   verdict")
    for k in keys:
        r_raw = spearmanr(df[f"gt_{k}"], df[f"deg_{k}"]).statistic
        r_sm = spearmanr(df[f"gt_{k}"], df[f"sm_{k}"]).statistic
        best = max(abs(r_raw), abs(r_sm))
        verdict = (
            "usable as-is" if abs(r_raw) > 0.7
            else "usable, smooth first" if abs(r_sm) > 0.7
            else "noise-dominated" if best < 0.4
            else "weak"
        )
        print(f"{k:<12} {r_raw:14.3f} {r_sm:16.3f}   {verdict}")

    print(
        "\nDecision rule:"
        "\n  PC1 features (grad_mean/var_med/var_iqr) usable -> FiLM on a"
        "\n    continuous detail-density score is well-founded."
        "\n  PC1 features noise-dominated but bg_frac/p50 survive -> condition on"
        "\n    the sparsity axis plus an explicit speckle-severity estimate"
        "\n    (FFDNet-style); drop the detail-density term."
        "\n  Nothing survives -> skip conditioning entirely, spend the budget on"
        "\n    PC1-stratified oversampling of the high-detail tail instead."
    )


if __name__ == "__main__":
    main()
