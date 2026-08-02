"""
Regime analysis for the KLA restoration challenge.

Question this answers: are 'texture' and 'dendrite' actually two regimes, or
are there more (or fewer) real clusters in the data? Drives the decision on
whether category-aware conditioning earns its complexity -- and, if so, which
statistics are informative enough to serve as the conditioning vector at
inference time on unseen sources.

Usage:
    python regime_analysis.py --gt_dir /path/to/ground_truth [--k_max 6]

Deliberately uses only statistics computable from a *degraded input* at test
time, so anything that separates clusters here is a candidate conditioning
feature -- no labels required.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def image_stats(arr: np.ndarray) -> dict:
    """Cheap, label-free descriptors of an image's signal regime."""
    x = arr.astype(np.float32)
    # Handle both 0-255 (uint8) and 0-1 (float32) ranges
    if x.max() > 1.5:
        x = x / 255.0

    # Sparsity: dendrite images are mostly near-zero background.
    bg_frac = float((x < 0.05).mean())

    # Dynamic range / contrast character.
    p1, p50, p99 = np.percentile(x, [1, 50, 99])

    # Gradient energy: dense granularity vs. sparse high-contrast edges.
    gy, gx = np.gradient(x)
    grad = np.sqrt(gx**2 + gy**2)
    grad_mean = float(grad.mean())
    # Kurtosis-like: sparse strong edges -> heavy tail; texture -> uniform.
    grad_tail = float(np.percentile(grad, 99) / (grad_mean + 1e-8))

    # Local variance distribution (proxy for speckle severity / homogeneity).
    k = 8
    h, w = x.shape
    hh, ww = (h // k) * k, (w // k) * k
    tiles = x[:hh, :ww].reshape(hh // k, k, ww // k, k).transpose(0, 2, 1, 3)
    tiles = tiles.reshape(-1, k * k)
    tile_var = tiles.var(axis=1)
    var_med = float(np.median(tile_var))
    var_iqr = float(np.percentile(tile_var, 75) - np.percentile(tile_var, 25))

    # Radial spectral slope: how energy falls off with frequency.
    f = np.fft.fftshift(np.abs(np.fft.fft2(x - x.mean())))
    cy, cx = f.shape[0] // 2, f.shape[1] // 2
    yy, xx = np.ogrid[: f.shape[0], : f.shape[1]]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    rmax = min(cy, cx)
    radial = np.array([f[r == i].mean() for i in range(1, rmax)])
    logr = np.log(np.arange(1, rmax))
    slope = float(np.polyfit(logr, np.log(radial + 1e-8), 1)[0])

    return {
        "bg_frac": bg_frac,
        "p1": float(p1),
        "p50": float(p50),
        "p99": float(p99),
        "grad_mean": grad_mean,
        "grad_tail": grad_tail,
        "var_med": var_med,
        "var_iqr": var_iqr,
        "spec_slope": slope,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--k_max", type=int, default=6)
    ap.add_argument("--out", default="regime_report.csv")
    args = ap.parse_args()

    paths = sorted(
        p for p in Path(args.gt_dir).rglob("*")
        if p.suffix.lower() in {".npy", ".png", ".tif", ".tiff", ".jpg", ".bmp"}
    )
    if not paths:
        raise SystemExit(f"No images found under {args.gt_dir}")

    rows, names = [], []
    for p in paths:
        if p.suffix.lower() == ".npy":
            arr = np.load(p)
            # Ensure it's 2D (grayscale); if 3D, convert to grayscale
            if arr.ndim == 3:
                arr = np.mean(arr, axis=2)
            # Ensure uint8 range; if already normalized, scale back
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        else:
            arr = np.array(Image.open(p).convert("L"))
        s = image_stats(arr)
        rows.append(list(s.values()))
        names.append(str(p.relative_to(args.gt_dir)))
    keys = list(image_stats(np.zeros((16, 16), np.uint8)).keys())

    X = np.array(rows, dtype=np.float64)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)

    # How many real regimes? Silhouette across k.
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    print(f"\n{len(paths)} images, {X.shape[1]} features\n")
    print("k   silhouette   cluster sizes")
    best = None
    for k in range(2, args.k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xz)
        sil = silhouette_score(Xz, km.labels_)
        sizes = np.bincount(km.labels_, minlength=k)
        print(f"{k}   {sil:8.3f}   {list(sizes)}")
        if best is None or sil > best[1]:
            best = (k, sil, km)

    k, sil, km = best
    print(f"\nBest k = {k} (silhouette {sil:.3f})")

    # Which features actually separate the clusters? These are the ones worth
    # feeding to FiLM conditioning; flat ones are dead weight.
    print("\nfeature          between-cluster spread (z-units)")
    centers = km.cluster_centers_
    for i, name in enumerate(keys):
        spread = centers[:, i].max() - centers[:, i].min()
        flag = "  <-- informative" if spread > 1.5 else ""
        print(f"{name:<15} {spread:6.2f}{flag}")

    with open(args.out, "w") as f:
        f.write("path,cluster," + ",".join(keys) + "\n")
        for n, lab, row in zip(names, km.labels_, X):
            f.write(f"{n},{lab}," + ",".join(f"{v:.6f}" for v in row) + "\n")
    print(f"\nWrote per-image assignments to {args.out}")

    print(
        "\nRead it like this:"
        "\n  silhouette peaks at k=2  -> two regimes, matches the given labels;"
        "\n                              conditioning is a coin flip, decide on"
        "\n                              the specialist fine-tune test."
        "\n  silhouette peaks at k>2  -> the label taxonomy is coarser than the"
        "\n                              data; conditioning likely pays, and the"
        "\n                              extra clusters are what the OOD test set"
        "\n                              is probably drawing from."
        "\n  no clear peak, all low   -> one continuous regime; skip conditioning"
        "\n                              and spend the complexity budget elsewhere."
    )


if __name__ == "__main__":
    main()
