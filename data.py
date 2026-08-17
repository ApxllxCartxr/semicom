from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from preprocessing import log_transform, synthesize_lr, downsample2, estimate_L
from regime_analysis import image_stats


def _load_npy(p: Path) -> np.ndarray:
    a = np.load(p).astype(np.float32)
    return np.squeeze(a)


def load_paired_dataset(gt_dir, deg_dir, gt_range=(0.0, 1.0), deg_max=5.0):
    gt_paths = sorted(Path(gt_dir).glob("*.npy"))
    if not gt_paths:
        raise ValueError(f"No .npy under {gt_dir}")
    deg_paths = []
    for gp in gt_paths:
        matches = sorted(Path(deg_dir).glob(gp.stem + ".*"))
        if not matches:
            raise ValueError(f"No degraded match for {gp.stem} in {deg_dir}")
        deg_paths.append(matches[0])
    # Assert range on a sample (0-1 GT; degraded >=0, may exceed 1 but bounded)
    for gp, dp in list(zip(gt_paths, deg_paths))[:8]:
        g, d = _load_npy(gp), _load_npy(dp)
        assert g.min() >= gt_range[0] - 1e-4 and g.max() <= gt_range[1] + 1e-4, \
            f"{gp.name}: GT range [{g.min():.3f},{g.max():.3f}] not in {gt_range}"
        # Degraded may dip slightly negative (~-0.1, processing artifacts) and exceed 1
        # (multiplicative speckle); log_transform clips the floor at eps downstream.
        assert d.min() >= -0.2 and d.max() <= deg_max, \
            f"{dp.name}: degraded range [{d.min():.3f},{d.max():.3f}] out of [-0.2,{deg_max}]"
    return gt_paths, deg_paths


def estimate_L_range(deg_paths, cfg, sample: int = 300):
    step = max(1, len(deg_paths) // sample)
    Ls = [estimate_L(_load_npy(p), domain=cfg.cond_domain) for p in deg_paths[::step]]
    lo, hi = float(np.percentile(Ls, 5)), float(np.percentile(Ls, 95))
    m = cfg.l_range_margin
    return (max(0.1, lo * (1 - m)), hi * (1 + m))


def compute_pc_features(gt_paths):
    from sklearn.decomposition import PCA
    feats = []
    for p in gt_paths:
        s = image_stats(_load_npy(p))               # image_stats handles 0-1
        feats.append([s["grad_mean"], s["var_med"], s["var_iqr"]])
    feats = np.asarray(feats, np.float64)
    z = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
    pcs = PCA(n_components=2, random_state=0).fit_transform(z)
    return pcs[:, 0], pcs[:, 1], feats


def stratified_split(pc1, pc2, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    q1 = np.digitize(pc1, np.percentile(pc1, [25, 50, 75]))
    q2 = np.digitize(pc2, np.percentile(pc2, [25, 50, 75]))
    strata = q1 * 4 + q2                              # 16 strata
    train_idx, val_idx = [], []
    for s in range(16):
        members = np.where(strata == s)[0]
        if len(members) == 0:
            continue
        rng.shuffle(members)
        n_val = max(1, int(round(len(members) * val_ratio)))
        val_idx.extend(members[:n_val].tolist())
        train_idx.extend(members[n_val:].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(val_idx)), strata


class TrainDataset(Dataset):
    """
    Each item: with prob synth_prob build LR by re-degradation (downsample+speckle at
    sampled L); else use the real provided NoisyLR. Crops are 2x-aligned:
    GT crop size c, LR crop size c//2 at the matching location.
    Conditioning is L̂ estimated FROM THE LR INPUT (train==test).
    """
    def __init__(self, cfg, gt_paths, deg_paths, indices, strata, L_range):
        self.cfg = cfg
        self.gt = [gt_paths[i] for i in indices]
        self.deg = [deg_paths[i] for i in indices]
        self.strata = strata[indices]
        self.L_range = L_range

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, i):
        cfg = self.cfg
        rng = np.random.default_rng()
        hr = _load_npy(self.gt[i])                       # (H,H) in 0-1
        H = hr.shape[0]
        c = cfg.crop_gt_512 if H >= 512 else cfg.crop_gt_256
        cl = c // cfg.sr_scale

        use_synth = rng.random() < cfg.synth_prob
        if use_synth:
            L = rng.uniform(*self.L_range)
            lr_full = synthesize_lr(hr, L, eps_clip=cfg.log_eps, rng=rng)   # (H/2,H/2)
        else:
            lr_full = _load_npy(self.deg[i])                                 # (H/2,H/2) real

        Hl = lr_full.shape[0]
        # aligned crop: choose LR top-left, map to HR by *2
        ly = rng.integers(0, Hl - cl + 1)
        lx = rng.integers(0, Hl - cl + 1)
        lr = lr_full[ly:ly + cl, lx:lx + cl]
        hr_c = hr[2 * ly:2 * ly + c, 2 * lx:2 * lx + c]

        # Gaussian hedge on a small subset (applied in linear, on LR)
        if rng.random() < cfg.gaussian_prob:
            sigma = rng.uniform(*cfg.gaussian_sigma_range)
            lr = np.clip(lr + rng.normal(0, sigma, lr.shape), cfg.log_eps, None)

        # flips + 90/180/270 (same k for both, structure orientation-free)
        if rng.random() < 0.5:
            lr, hr_c = np.flipud(lr).copy(), np.flipud(hr_c).copy()
        if rng.random() < 0.5:
            lr, hr_c = np.fliplr(lr).copy(), np.fliplr(hr_c).copy()
        k = int(rng.integers(0, 4))
        lr, hr_c = np.rot90(lr, k).copy(), np.rot90(hr_c, k).copy()

        log_L = float(np.log(estimate_L(lr, domain=cfg.cond_domain)))       # L̂ from INPUT
        lr_log = log_transform(lr, cfg.log_eps)
        hr_log = log_transform(np.clip(hr_c, cfg.log_eps, None), cfg.log_eps)
        return {
            "input": torch.from_numpy(lr_log)[None],
            "target": torch.from_numpy(hr_log)[None],
            "log_L": torch.tensor(log_L, dtype=torch.float32),
            "stratum": int(self.strata[i]),
        }


class ValDataset(Dataset):
    """Deterministic: real provided pairs, full-frame (no crop/aug). Carries stratum
    for per-bin reporting. Conditioning is L̂ from the real LR input."""
    def __init__(self, cfg, gt_paths, deg_paths, indices, strata):
        self.cfg = cfg
        self.gt = [gt_paths[i] for i in indices]
        self.deg = [deg_paths[i] for i in indices]
        self.strata = strata[indices]

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, i):
        cfg = self.cfg
        hr = np.clip(_load_npy(self.gt[i]), cfg.log_eps, None)
        lr = np.clip(_load_npy(self.deg[i]), cfg.log_eps, None)
        log_L = float(np.log(estimate_L(lr, domain=cfg.cond_domain)))
        return {
            "input": torch.from_numpy(log_transform(lr, cfg.log_eps))[None],
            "target": torch.from_numpy(log_transform(hr, cfg.log_eps))[None],
            "log_L": torch.tensor(log_L, dtype=torch.float32),
            "stratum": int(self.strata[i]),
        }


def make_train_loader(cfg, ds: "TrainDataset", pc1_train: np.ndarray) -> DataLoader:
    thr = np.percentile(pc1_train, 90)                 # top decile
    weights = np.where(pc1_train >= thr, cfg.pc1_oversample_strength, 1.0).astype(np.float64)
    sampler = WeightedRandomSampler(weights, num_samples=len(ds) * cfg.crops_per_image,
                                    replacement=True)
    return DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler,
                      num_workers=cfg.num_workers, pin_memory=True, drop_last=True)


# ---- Tests (run directly, not via pytest) ----

def test_split_disjoint():
    rng = np.random.default_rng(0)
    pc1, pc2 = rng.standard_normal(500), rng.standard_normal(500)
    tr, va, st = stratified_split(pc1, pc2, 0.2, 0)
    assert len(np.intersect1d(tr, va)) == 0
    assert len(tr) + len(va) == 500
    assert 0.15 * 500 <= len(va) <= 0.25 * 500


def test_train_item_shapes():
    import tempfile, os
    from config import Config
    cfg = Config(); cfg.synth_prob = 1.0
    hr = (np.random.default_rng(0).random((256, 256))).astype(np.float32)
    d = tempfile.mkdtemp(); np.save(os.path.join(d, "000.npy"), hr)
    gp = [Path(d) / "000.npy"]; dp = gp  # deg unused when synth_prob=1
    ds = TrainDataset(cfg, gp, dp, np.array([0]), np.array([0]), (2.0, 6.0))
    item = ds[0]
    assert item["input"].shape == (1, 64, 64)
    assert item["target"].shape == (1, 128, 128)
    assert item["log_L"].ndim == 0
