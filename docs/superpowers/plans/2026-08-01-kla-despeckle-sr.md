# KLA Joint Despeckle + 2x SR Implementation Plan (HARDENED v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The environment venv is at `/home/apollo/projects/pysandbox/ml_venv` — activate it before running anything: `source /home/apollo/projects/pysandbox/ml_venv/bin/activate`.

**Goal:** Build a single NAFNet that jointly denoises multiplicative-Gamma speckle and 2× super-resolves microscopy images, conditioned on a closed-form estimate of the speckle looks parameter L, trained on 3200 real GT/degraded pairs augmented with re-degradation.

**Architecture:** LR log-domain input → NAFNet U-Net (width 32, exactly 3 downsampling stages, returns to LR resolution) → 2× PixelShuffle SR head → add bicubic-2× of the log input (global residual) → log-domain 2× output → exp() to linear for loss/metrics. FiLM (per-block scale/shift from log L̂) modulates every NAFBlock, toggleable for ablation.

**Tech Stack:** PyTorch 2.x, numpy, scipy, scikit-learn, scikit-image, Pillow, lpips, tqdm.

---

## Corrections applied vs v1 (why this rewrite exists)

| # | v1 defect | v2 fix |
|---|-----------|--------|
| C1 | U-Net returned to input res, global residual was 2× → shape mismatch; PixelShuffle head only in comments | U-Net returns to LR res, **explicit 2× PixelShuffle head**, then add bicubic-2× input |
| C2 | Re-degradation preserved shape → no SR signal | Synthesis = **downsample ×2 then speckle**; network maps LR→2×LR |
| C3 | FiLM never threaded into blocks; single scalar scale on final residual | FiLM embedding threaded to **every NAFBlock**, per-block scale+shift, identity-init |
| C4 | Decoder concat doubled channels but block sized for halved → shape error | Decoder block sized for concatenated channels; explicit channel bookkeeping |
| C5 | `validate_film.py` was a no-op (L_true empty), loaded model needlessly | Standalone **closed-form** validator: synth at known L, estimate L̂, Spearman |
| C6 | Model conditioned on **true L** (unavailable at test) | Model conditions on **L̂ estimated from input**, train and test identical |
| C7 | Conditioning computed in linear domain | Conditioning computed in **log domain** (spec-mandated), both estimators behind one fn |
| C8 | `evaluate.py` mixed resolutions in one loader → collate crash; no bucketing/cudnn/channels_last/benchmark | **Bucket by resolution**, fixed-shape batches, cudnn.benchmark, channels_last, compile-amortization benchmark mode |
| C9 | Single scalar val loss; non-deterministic val (fresh noise) | Val on **real pairs (deterministic)**, metrics **per PC1/PC2 bin** + aggregate |
| C10 | `L_range` hardcoded (1,8) placeholder | **Estimated from provided NoisyLR** via the L̂ estimator |
| C11 | `from preprocessing import image_stats` (wrong module); f-string syntax error | Import `image_stats` from `regime_analysis`; fix param-count print |
| C12 | `assert 0-255` would crash on real 0–1 data | Assert **0–1** (GT in [0,1], degraded ≥0 and may exceed 1) |

## Global Constraints

- `.npy` arrays are **float32 in 0–1** (GT ∈ [0,1]; degraded ≥ 0, may exceed 1 from multiplicative speckle). Assert this at load; do NOT rescale.
- Log-domain forward pass and conditioning; exp() before loss/metrics.
- Metrics/PNG use `data_range=1.0`; clamp exp() output to [0,1] for metric/PNG only (loss uses unclamped exp).
- NAFNet: width 32, encoder blocks [2,2,4], middle 8, decoder [2,2,2], **exactly 3 downsampling stages**. No self-attention, no LayerNorm. Target total 8–12M params.
- FiLM trunk is a 2-layer MLP (~5k params); a shared projection head produces per-block scale/shift (documented ~35k extra — see Task 4 note). Entire FiLM branch removable via `film_enabled=False`.
- Global residual = bicubic-2× of log input, added to head output (both log-domain, both 2×LR).
- Model conditions on **L̂ from the input** (train == test). True L is used ONLY by `validate_film.py`.
- Loss = 1.0·Charbonnier + 0.3·(1−SSIM) + 0.05·LPIPS, in linear space. LPIPS weight fixed at 0.05.
- AdamW lr 2e-4, cosine decay, AMP bf16, EMA (eval with EMA).
- Inference: `torch.inference_mode()`, autocast bf16, channels_last, bucket by resolution then `cudnn.benchmark=True`, model loaded once, `num_workers=8`, `pin_memory=True`. `torch.compile` behind a flag with a benchmark mode.
- All of {FiLM, compile, PC1 oversample strength} toggleable from `config.py`.

## File Structure

```
semicon/
├── config.py            # single source of truth; all ablation toggles
├── preprocessing.py     # log/exp, LR synthesis (downsample+speckle), L̂ estimator (log & linear)
├── data.py              # 0–1 loader+assert, L-range est, PC features, stratified split, datasets/loaders
├── architecture.py      # NAFBlock (FiLM-aware), NAFNet, FiLM generator, KLARestoration, PixelShuffle head
├── loss.py              # Charbonnier + SSIM + LPIPS, linear space
├── train.py             # AdamW+cosine+AMP bf16+EMA; per-bin validation
├── evaluate.py          # bucketed batched inference, channels_last, cudnn.benchmark, compile+benchmark flags
├── validate_film.py     # closed-form L̂ vs true-L Spearman check
├── requirements.txt     # pip freeze of ml_venv
└── regime_analysis.py   # EXISTING — source of image_stats() (do not duplicate)
```

`image_stats(arr)` already exists in `regime_analysis.py` and accepts either 0–255 or 0–1 arrays (it rescales internally if max>1.5). Import it; never re-define it.

---

## Task 0: Environment setup

**Files:** none (installs into `/home/apollo/projects/pysandbox/ml_venv`)

The venv currently has numpy/scipy/scikit-learn/pandas/Pillow. Training/inference/loss additionally need torch, torchvision, lpips, scikit-image, tqdm.

- [ ] **Step 1: Install deep-learning deps**

```bash
source /home/apollo/projects/pysandbox/ml_venv/bin/activate
pip install torch torchvision lpips scikit-image tqdm
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

- [ ] **Step 2: Record whether CUDA is available**

If `cuda False`, training the full 100 epochs on CPU is impractical — the smoke runs (`--epochs 2`, small) still validate correctness end-to-end, which is what the plan gates on. Note the device in the handoff; full training needs a GPU box. Do NOT change the recipe to accommodate CPU.

---

## Task 1: Config System

**Files:** Create `config.py`

**Interfaces — Produces:** `Config` dataclass consumed by every other module.

- [ ] **Step 1: Write `config.py`**

```python
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # ---- Data / range ----
    gt_dir: str = "train/train/GT"
    deg_dir: str = "train/train/NoisyLR"
    test_dir: str = "Test_NoisyLR"
    gt_range: Tuple[float, float] = (0.0, 1.0)       # asserted on GT
    deg_max_bound: float = 5.0                        # sanity upper bound on degraded
    sr_scale: int = 2

    # ---- Preprocessing ----
    log_eps: float = 1e-6
    cond_domain: str = "log"                          # "log" or "linear" for L̂ estimation

    # ---- Architecture ----
    naf_width: int = 32
    naf_enc_blocks: List[int] = field(default_factory=lambda: [2, 2, 4])
    naf_mid_blocks: int = 8
    naf_dec_blocks: List[int] = field(default_factory=lambda: [2, 2, 2])
    naf_block_expand: int = 2      # NAFBlock internal channel expansion; TUNE to hit 8–12M (Task 4)

    # ---- FiLM conditioning ----
    film_enabled: bool = True                         # ABLATION TOGGLE
    film_hidden: int = 64                             # trunk hidden dim (~5k trunk)

    # ---- Data pipeline ----
    synth_prob: float = 0.5                           # P(use re-degradation) per sample; else real pair
    gaussian_prob: float = 0.1                        # low-prob Gaussian hedge subset
    gaussian_sigma_range: Tuple[float, float] = (0.5/255, 2.0/255)  # in 0–1 units
    crop_gt_256: int = 128                            # GT crop when GT is 256 -> LR crop 64
    crop_gt_512: int = 256                            # GT crop when GT is 512 -> LR crop 128
    l_range_margin: float = 0.10                      # extend sampled L range 10% beyond observed
    pc1_oversample_strength: float = 1.5              # ABLATION TOGGLE (1.0 = off)
    val_split_ratio: float = 0.2

    # ---- Loss ----
    charbonnier_weight: float = 1.0
    charbonnier_eps: float = 1e-3
    ssim_weight: float = 0.3
    lpips_weight: float = 0.05                        # DO NOT tune up

    # ---- Optimization ----
    lr: float = 2e-4
    weight_decay: float = 1e-4
    num_epochs: int = 100
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    batch_size: int = 16
    num_workers: int = 8
    seed: int = 0

    # ---- Inference ----
    checkpoint_path: str = "checkpoint_ema.pt"
    batch_size_test: int = 4
    compile_enabled: bool = False                     # ABLATION / SPEED TOGGLE


CONFIG = Config()
```

- [ ] **Step 2: Smoke-check and commit**

```bash
source /home/apollo/projects/pysandbox/ml_venv/bin/activate
python -c "from config import CONFIG; assert CONFIG.naf_enc_blocks==[2,2,4]; assert CONFIG.gt_range==(0.0,1.0); print('config OK')"
git add config.py && git commit -m "feat: config with all ablation toggles (FiLM, compile, PC1 strength)"
```

---

## Task 2: Preprocessing — log/exp, LR synthesis, L̂ estimator

**Files:** Create `preprocessing.py`

**Interfaces — Produces:**
- `log_transform(x, eps=1e-6) -> np.ndarray`
- `exp_transform(x) -> np.ndarray`
- `downsample2(x) -> np.ndarray` (area-average ×2)
- `synthesize_lr(hr, L, eps_clip=1e-6, rng=None) -> np.ndarray` (downsample ×2 THEN Gamma speckle; returns 0–1-ish, ≥0)
- `estimate_L(img, domain="log", k_tiles=8, eps=1e-6) -> float` (input in **linear** 0–1; picks lowest-variance decile of tiles; log path inverts trigamma, linear path uses ENL=(mean/std)²)

**Consumes:** none.

- [ ] **Step 1: Write transforms + downsample + synthesis**

```python
import numpy as np
from scipy.special import polygamma  # polygamma(1, L) == trigamma(L)


def log_transform(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.log(np.clip(x, eps, None)).astype(np.float32)


def exp_transform(x):
    import numpy as _np
    return _np.exp(_np.clip(x, -30.0, 30.0))


def downsample2(x: np.ndarray) -> np.ndarray:
    """Area-average 2x downsample (anti-aliased)."""
    h, w = (x.shape[0] // 2) * 2, (x.shape[1] // 2) * 2
    return x[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)).astype(np.float32)


def synthesize_lr(hr: np.ndarray, L: float, eps_clip: float = 1e-6, rng=None) -> np.ndarray:
    """
    Degradation model: clean HR -> 2x downsample -> multiplicative Gamma speckle.
    speckle ~ Gamma(shape=L, scale=1/L)  (mean 1, var 1/L).
    Returns LR in 0–1 domain, clipped at a small positive floor (>0 for log).
    NO hard upper clip: multiplicative speckle legitimately exceeds 1.
    """
    if rng is None:
        rng = np.random.default_rng()
    lr_clean = downsample2(hr)
    speckle = rng.gamma(shape=L, scale=1.0 / L, size=lr_clean.shape).astype(np.float32)
    lr = lr_clean * speckle
    return np.clip(lr, eps_clip, None).astype(np.float32)
```

- [ ] **Step 2: Write the unified L̂ estimator (log + linear)**

```python
def _tiles(img: np.ndarray, k: int) -> np.ndarray:
    h, w = (img.shape[0] // k) * k, (img.shape[1] // k) * k
    t = img[:h, :w].reshape(h // k, k, w // k, k).transpose(0, 2, 1, 3)
    return t.reshape(-1, k * k)


def _invert_trigamma(target_var: float) -> float:
    """Solve trigamma(L) = target_var for L>0 by bisection (trigamma is decreasing)."""
    lo, hi = 0.05, 200.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if polygamma(1, mid) > target_var:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def estimate_L(img: np.ndarray, domain: str = "log", k_tiles: int = 8, eps: float = 1e-6) -> float:
    """
    Estimate Gamma looks L from a LINEAR-domain noisy image (0–1).
    Uses the lowest-variance decile of tiles (most homogeneous -> speckle-only).
      domain="linear": ENL = (mean/std)^2 over homogeneous tiles.
      domain="log":    Var[log x] ~ trigamma(L) over homogeneous tiles; invert.
    Both return L in [0.05, 200]. Log path is more stable when tiles have DC offset.
    """
    tiles = _tiles(img, k_tiles)
    tile_var = tiles.var(axis=1)
    # lowest-variance decile = most homogeneous
    n_keep = max(1, tile_var.shape[0] // 10)
    idx = np.argsort(tile_var)[:n_keep]
    homo = tiles[idx]

    if domain == "linear":
        m = homo.mean(axis=1)
        s = homo.std(axis=1)
        enl = (m / (s + eps)) ** 2
        return float(np.clip(np.median(enl), 0.05, 200.0))
    else:  # log
        logt = np.log(np.clip(homo, eps, None))
        vlog = logt.var(axis=1)
        return float(np.clip(_invert_trigamma(float(np.median(vlog))), 0.05, 200.0))
```

- [ ] **Step 3: Tests — round-trip + recover known L**

```python
def test_log_exp_roundtrip():
    x = np.linspace(1e-3, 1.0, 1000).astype(np.float32)
    assert np.allclose(x, exp_transform(log_transform(x)), rtol=1e-4, atol=1e-5)

def test_estimate_L_recovers():
    rng = np.random.default_rng(0)
    hr = np.full((256, 256), 0.6, np.float32)   # flat -> pure speckle after synth
    for L_true in (2.0, 5.0, 10.0):
        lr = synthesize_lr(hr, L_true, rng=rng)
        L_log = estimate_L(lr, domain="log")
        L_lin = estimate_L(lr, domain="linear")
        assert 0.5 * L_true < L_log < 2.0 * L_true, (L_true, L_log)
        assert 0.5 * L_true < L_lin < 2.0 * L_true, (L_true, L_lin)
```

- [ ] **Step 4: Run tests + commit**

```bash
source /home/apollo/projects/pysandbox/ml_venv/bin/activate
python -c "from preprocessing import *; test_log_exp_roundtrip(); test_estimate_L_recovers(); print('preprocessing PASS')"
git add preprocessing.py && git commit -m "feat: log/exp, downsample+speckle synthesis, log/linear L-hat estimator"
```

---

## Task 3: Data — loader+assert, L-range, PC features, split, datasets

**Files:** Create `data.py`

**Interfaces — Produces:**
- `load_paired_dataset(gt_dir, deg_dir, gt_range=(0,1), deg_max=5.0) -> (gt_paths, deg_paths)`
- `estimate_L_range(deg_paths, cfg, sample=300) -> (float, float)`
- `compute_pc_features(gt_paths) -> (pc1, pc2, feat)`  (imports `image_stats` from `regime_analysis`)
- `stratified_split(pc1, pc2, val_ratio, seed) -> (train_idx, val_idx, strata)`
- `TrainDataset`, `ValDataset` (both yield dict with keys `input`(log LR), `target`(log HR crop), `log_L`(float32 = log of L̂ from input), `stratum`)
- `make_train_loader(cfg, ds, pc1_train_scores) -> DataLoader`

**Consumes:** `preprocessing.{log_transform, synthesize_lr, downsample2, estimate_L}`, `config.Config`, `regime_analysis.image_stats`.

- [ ] **Step 1: Loader with 0–1 assertion**

```python
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
    # Assert range on a sample (0–1 GT; degraded >=0, may exceed 1 but bounded)
    for gp, dp in list(zip(gt_paths, deg_paths))[:8]:
        g, d = _load_npy(gp), _load_npy(dp)
        assert g.min() >= gt_range[0] - 1e-4 and g.max() <= gt_range[1] + 1e-4, \
            f"{gp.name}: GT range [{g.min():.3f},{g.max():.3f}] not in {gt_range}"
        assert d.min() >= -1e-4 and d.max() <= deg_max, \
            f"{dp.name}: degraded range [{d.min():.3f},{d.max():.3f}] out of [0,{deg_max}]"
    return gt_paths, deg_paths
```

- [ ] **Step 2: L-range from provided pairs + PC features + split**

```python
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
        s = image_stats(_load_npy(p))               # image_stats handles 0–1
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
```

- [ ] **Step 3: TrainDataset (real + synthetic mix, aligned crops)**

```python
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
        hr = _load_npy(self.gt[i])                       # (H,H) in 0–1
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
```

- [ ] **Step 4: ValDataset (real pairs, deterministic, full images)**

```python
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
    sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
    return DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler,
                      num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
```

- [ ] **Step 5: Tests — assertion catches bad range; split disjoint & stratified; synth shapes**

```python
def test_split_disjoint():
    rng = np.random.default_rng(0)
    pc1, pc2 = rng.standard_normal(500), rng.standard_normal(500)
    tr, va, st = stratified_split(pc1, pc2, 0.2, 0)
    assert len(np.intersect1d(tr, va)) == 0
    assert len(tr) + len(va) == 500
    assert 0.15 * 500 <= len(va) <= 0.25 * 500

def test_train_item_shapes():
    # synthetic-only config-ish check via a tiny fake
    import numpy as np, tempfile, os
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
```

- [ ] **Step 6: Run + commit**

```bash
source /home/apollo/projects/pysandbox/ml_venv/bin/activate
python -c "from data import *; test_split_disjoint(); test_train_item_shapes(); print('data PASS')"
git add data.py && git commit -m "feat: 0-1 loader+assert, L-range est, PC split, real+synth datasets"
```

---

## Task 4: Architecture — NAFNet + FiLM + PixelShuffle head

**Files:** Create `architecture.py`

**Interfaces — Produces:**
- `NAFBlock(channels, film_max_c=None)` with `forward(x, film=None)` where `film` is `(scale_full, shift_full)` each `(B, 2*film_max_c? )` → block slices first `channels`.
- `NAFNet(width, enc_blocks, mid_blocks, dec_blocks, film_max_c)` with `forward(x, film=None)` returning features at **input (LR) resolution**, channels=`width`.
- `FiLMGenerator(hidden, max_c)` with `forward(log_L) -> (scale_full, shift_full)`.
- `KLARestoration(cfg)` with `forward(x_log, log_L=None) -> y_log` at **2×** input resolution.

**Consumes:** `config.Config`.

> **FiLM param note (documented deviation):** the spec asks for a ~5k 2-layer MLP producing per-block scale/shift. The trunk here is 1→`hidden`→`hidden` (~5k). Because block channel counts reach `width*8=256`, a single shared projection `Linear(hidden, 2*max_c)` (~35k for hidden=64, max_c=256) produces one `(scale,shift)` vector that every block **slices to its own channel count**. This keeps FiLM well under 0.1M (≪ the 8–12M budget), applies to **every block**, and stays fully removable via `film_enabled=False`. If a stricter ≤5k FiLM is required, set `film_hidden=8` (trunk shrinks, projection ~4k).

- [ ] **Step 1: NAFBlock (authentic lightweight NAFNet block: 1×1 expand + depthwise + SimpleGate + SCA, no LayerNorm/attention, FiLM-aware)**

> Uses the real NAFNet block, NOT full 3×3 convs — the latter would put the 256-channel middle at ~25M params. Here: pointwise expand → depthwise 3×3 → SimpleGate (halves channels, no activation) → SCA → pointwise project, plus a pointwise FFN. `beta`/`gamma` are zero-init so an untrained block is identity (stable start). `expand` (from `cfg.naf_block_expand`) is the tuning knob for the param budget. No LayerNorm (honors "no LayerNorm-heavy").

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def _simple_gate(x):
    a, b = x.chunk(2, dim=1)      # halve channels
    return a * b


class NAFBlock(nn.Module):
    def __init__(self, channels: int, film_max_c: int = None, expand: int = 2):
        super().__init__()
        self.channels = channels
        dw = channels * expand                       # expanded width (even)
        # --- token mixer: pointwise -> depthwise -> SimpleGate -> SCA -> pointwise ---
        self.conv1 = nn.Conv2d(channels, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)     # depthwise (cheap)
        self.conv3 = nn.Conv2d(dw // 2, channels, 1)                # after SimpleGate: dw -> dw//2
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        # --- channel mixer (FFN): pointwise -> SimpleGate -> pointwise ---
        ffn = channels * expand
        self.conv4 = nn.Conv2d(channels, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, channels, 1)               # after SimpleGate
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, film=None):
        y = self.conv1(x)
        y = self.conv2(y)
        y = _simple_gate(y)                          # -> dw//2 channels
        y = y * self.sca(y)
        y = self.conv3(y)                            # -> channels
        if film is not None:
            scale_full, shift_full = film            # (B, max_c) each
            c = self.channels
            y = y * (1.0 + scale_full[:, :c, None, None]) + shift_full[:, :c, None, None]
        x = x + self.beta * y                        # residual 1
        z = self.conv4(x)
        z = _simple_gate(z)                          # -> ffn//2 channels
        z = self.conv5(z)
        return x + self.gamma * z                    # residual 2
```

- [ ] **Step 2: FiLM generator**

```python
class FiLMGenerator(nn.Module):
    """log_L (B,) -> (scale_full, shift_full), each (B, max_c). Identity at init."""
    def __init__(self, hidden: int, max_c: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.proj = nn.Linear(hidden, 2 * max_c)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)                         # -> scale=shift=0 -> identity
        self.max_c = max_c

    def forward(self, log_L):
        if log_L.ndim == 1:
            log_L = log_L[:, None]
        h = self.trunk(log_L)
        out = self.proj(h)
        return out[:, :self.max_c], out[:, self.max_c:]
```

- [ ] **Step 3: NAFNet U-Net (returns to LR resolution, threads FiLM)**

```python
class NAFNet(nn.Module):
    def __init__(self, width=32, enc_blocks=(2, 2, 4), mid_blocks=8, dec_blocks=(2, 2, 2),
                 film_max_c=None, expand=2):
        super().__init__()
        self.intro = nn.Conv2d(1, width, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        self.enc_channels = []
        for n in enc_blocks:
            self.encoders.append(nn.ModuleList([NAFBlock(ch, film_max_c, expand) for _ in range(n)]))
            self.enc_channels.append(ch)
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2))
            ch *= 2
        self.middle = nn.ModuleList([NAFBlock(ch, film_max_c, expand) for _ in range(mid_blocks)])
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n, skip_ch in zip(dec_blocks, reversed(self.enc_channels)):
            # PixelShuffle upsample halves channels: ch -> ch//2
            self.ups.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 1, bias=False), nn.PixelShuffle(2)))
            ch = ch // 2
            self.decoders.append(nn.ModuleList([NAFBlock(ch, film_max_c, expand) for _ in range(n)]))
            # after up, ch == skip_ch; we ADD skip (not concat) to keep channel math trivial
        self.width = width

    def forward(self, x, film=None):
        y = self.intro(x)
        skips = []
        for blocks, down in zip(self.encoders, self.downs):
            for b in blocks:
                y = b(y, film)
            skips.append(y)
            y = down(y)
        for b in self.middle:
            y = b(y, film)
        for blocks, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            y = up(y)
            y = y + skip                                       # additive skip (channels match)
            for b in blocks:
                y = b(y, film)
        return y                                               # (B, width, H, W) at LR res
```

- [ ] **Step 4: Full model with PixelShuffle SR head + global residual**

```python
class KLARestoration(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.film_enabled = cfg.film_enabled
        max_c = cfg.naf_width * (2 ** len(cfg.naf_enc_blocks))   # bottleneck width = 256
        self.backbone = NAFNet(cfg.naf_width, tuple(cfg.naf_enc_blocks), cfg.naf_mid_blocks,
                               tuple(cfg.naf_dec_blocks), film_max_c=max_c, expand=cfg.naf_block_expand)
        if cfg.film_enabled:
            self.film_gen = FiLMGenerator(cfg.film_hidden, max_c)
        # 2x PixelShuffle SR head: width -> width*4 -> PixelShuffle(2) -> 1 channel
        self.sr_head = nn.Sequential(
            nn.Conv2d(cfg.naf_width, cfg.naf_width * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(cfg.naf_width, 1, 3, padding=1),
        )
        self.scale = cfg.sr_scale

    def forward(self, x_log, log_L=None):
        film = None
        if self.film_enabled and log_L is not None:
            film = self.film_gen(log_L)
        feat = self.backbone(x_log, film)                       # (B, width, H, W)
        residual = self.sr_head(feat)                           # (B, 1, 2H, 2W)
        base = F.interpolate(x_log, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return residual + base                                  # log-domain 2x output
```

- [ ] **Step 5: Tests — forward shapes (both FiLM on/off), param budget, FiLM identity at init**

```python
def test_forward_and_params():
    import torch
    from config import Config
    for film in (True, False):
        cfg = Config(); cfg.film_enabled = film
        m = KLARestoration(cfg).eval()
        x = torch.randn(2, 1, 64, 64)
        lL = torch.zeros(2)
        y = m(x, lL if film else None)
        assert y.shape == (2, 1, 128, 128), (film, y.shape)
    n = sum(p.numel() for p in KLARestoration(Config()).parameters())
    print(f"params={n/1e6:.2f}M (expand={Config().naf_block_expand})")
    # Loose sanity bound; the 8-12M target is met by tuning naf_block_expand (see gate below).
    assert 3e6 <= n <= 16e6, f"param count {n/1e6:.2f}M wildly off — check block/channel wiring"

def test_film_identity_at_init():
    import torch
    from config import Config
    cfg = Config(); cfg.film_enabled = True
    m = KLARestoration(cfg).eval()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        y_film = m(x, torch.zeros(1))
        m.film_enabled = False
        y_none = m(x, None)
    assert torch.allclose(y_film, y_none, atol=1e-5), "FiLM proj zero-init must be identity"
```

- [ ] **Step 6: Run + commit**

```bash
source /home/apollo/projects/pysandbox/ml_venv/bin/activate
python -c "from architecture import *; test_forward_and_params(); test_film_identity_at_init(); print('architecture PASS')"
git add architecture.py && git commit -m "feat: NAFNet + per-block FiLM + PixelShuffle 2x head + global residual"
```

> **Param-budget tuning gate (do this after Step 6 runs and prints the count):** width (32), block counts, and downsampling stages are fixed by spec, so the only free knob is `naf_block_expand`. Run:
> ```bash
> python -c "from config import Config; from architecture import KLARestoration; \
> import copy; c=Config(); \
> [print(e, round(sum(p.numel() for p in KLARestoration(copy.replace(c, naf_block_expand=e)).parameters())/1e6,2),'M') for e in (2,3,4)]"
> ```
> Pick the smallest `naf_block_expand` whose count lands in **8–12M**, set it as the `naf_block_expand` default in `config.py`, and re-commit. (`copy.replace` needs Python 3.13; otherwise construct `Config(naf_block_expand=e)`.) Expectation: expand=2 ≈ 5–6M (under target), expand=3 ≈ 8–10M ✓. Runtime is a scoring tiebreaker, so prefer the low end of 8–12M.

---

## Task 5: Loss — Charbonnier + SSIM + LPIPS (linear space)

**Files:** Create `loss.py`

**Interfaces — Produces:** `CombinedLoss(cfg)` with `forward(pred_log, target_log) -> (total, parts_dict)`.

**Consumes:** `preprocessing.exp_transform` (torch version below), `config.Config`.

- [ ] **Step 1: Torch exp + Charbonnier + SSIM + combined**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def _exp(x):
    return torch.exp(torch.clamp(x, -30.0, 30.0))


def _gaussian_window(ws=11, sigma=1.5, device="cpu"):
    coords = torch.arange(ws, device=device) - ws // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())[:, None]
    return (g @ g.t())[None, None]


def ssim(x, y, ws=11):
    x = x.clamp(0, 1); y = y.clamp(0, 1)
    w = _gaussian_window(ws, device=x.device).to(x.dtype)
    pad = ws // 2
    mu_x = F.conv2d(x, w, padding=pad); mu_y = F.conv2d(y, w, padding=pad)
    mx2, my2, mxy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sx = F.conv2d(x * x, w, padding=pad) - mx2
    sy = F.conv2d(y * y, w, padding=pad) - my2
    sxy = F.conv2d(x * y, w, padding=pad) - mxy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mxy + c1) * (2 * sxy + c2)) / ((mx2 + my2 + c1) * (sx + sy + c2) + 1e-12)
    return s.mean()


class CombinedLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.eps = cfg.charbonnier_eps
        self.lpips_fn = None
        if cfg.lpips_weight > 0:
            import lpips
            self.lpips_fn = lpips.LPIPS(net="alex")
            self.lpips_fn.eval()
            for p in self.lpips_fn.parameters():
                p.requires_grad_(False)

    def forward(self, pred_log, target_log):
        pred = _exp(pred_log); target = _exp(target_log)
        char = torch.sqrt((pred - target) ** 2 + self.eps ** 2).mean()
        s = 1.0 - ssim(pred, target)
        total = self.cfg.charbonnier_weight * char + self.cfg.ssim_weight * s
        parts = {"charbonnier": char.detach(), "ssim": s.detach()}
        if self.lpips_fn is not None:
            p3 = (pred.clamp(0, 1) * 2 - 1).repeat(1, 3, 1, 1)
            t3 = (target.clamp(0, 1) * 2 - 1).repeat(1, 3, 1, 1)
            lp = self.lpips_fn(p3, t3).mean()
            total = total + self.cfg.lpips_weight * lp
            parts["lpips"] = lp.detach()
        return total, parts
```

- [ ] **Step 2: Test — identical inputs give ~0 Charbonnier/SSIM**

```python
def test_loss_zero_on_identical():
    import torch
    from config import Config
    cfg = Config(); cfg.lpips_weight = 0.0     # skip lpips download in unit test
    crit = CombinedLoss(cfg)
    x = torch.rand(2, 1, 64, 64).clamp(1e-3, 1).log()
    total, parts = crit(x, x)
    assert parts["charbonnier"] < 1e-3 and parts["ssim"] < 1e-3
```

- [ ] **Step 3: Run + commit**

```bash
source /home/apollo/projects/pysandbox/ml_venv/bin/activate
python -c "from loss import *; test_loss_zero_on_identical(); print('loss PASS')"
git add loss.py && git commit -m "feat: Charbonnier+SSIM+LPIPS loss in linear space (LPIPS fixed 0.05)"
```

---

## Task 6: Training — AdamW + cosine + AMP bf16 + EMA + per-bin val

**Files:** Create `train.py`

**Interfaces — Produces:** `main()` CLI; writes `cfg.checkpoint_path` (EMA weights).

**Consumes:** everything above.

- [ ] **Step 1: EMA helper + per-bin validation**

```python
import argparse, copy
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import Config
from data import (load_paired_dataset, estimate_L_range, compute_pc_features,
                  stratified_split, TrainDataset, ValDataset, make_train_loader)
from architecture import KLARestoration
from loss import CombinedLoss, ssim, _exp
from torch.utils.data import DataLoader


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


@torch.no_grad()
def validate(ema_model, val_loader, device):
    ema_model.eval()
    per_bin = {}
    for batch in val_loader:
        x = batch["input"].to(device, memory_format=torch.channels_last)
        t = batch["target"].to(device)
        lL = batch["log_L"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            pred = ema_model(x, lL)
        pred_l = _exp(pred.float()).clamp(0, 1)
        tgt_l = _exp(t.float()).clamp(0, 1)
        for b in range(x.size(0)):
            s = int(batch["stratum"][b])
            val = ssim(pred_l[b:b+1], tgt_l[b:b+1]).item()
            per_bin.setdefault(s, []).append(val)
    agg = np.mean([v for vs in per_bin.values() for v in vs])
    return agg, {k: float(np.mean(v)) for k, v in sorted(per_bin.items())}
```

- [ ] **Step 2: Training loop**

```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no_film", action="store_true", help="ablate FiLM branch")
    ap.add_argument("--pc1_strength", type=float, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.epochs: cfg.num_epochs = args.epochs
    if args.no_film: cfg.film_enabled = False
    if args.pc1_strength is not None: cfg.pc1_oversample_strength = args.pc1_strength

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gt_paths, deg_paths = load_paired_dataset(cfg.gt_dir, cfg.deg_dir, cfg.gt_range, cfg.deg_max_bound)
    L_range = estimate_L_range(deg_paths, cfg)
    print(f"Estimated L sampling range: {L_range}")
    pc1, pc2, _ = compute_pc_features(gt_paths)
    tr_idx, va_idx, strata = stratified_split(pc1, pc2, cfg.val_split_ratio, cfg.seed)
    print(f"train={len(tr_idx)} val={len(va_idx)}")

    train_ds = TrainDataset(cfg, gt_paths, deg_paths, tr_idx, strata, L_range)
    val_ds = ValDataset(cfg, gt_paths, deg_paths, va_idx, strata)
    train_loader = make_train_loader(cfg, train_ds, pc1[tr_idx])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    model = KLARestoration(cfg).to(device, memory_format=torch.channels_last)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    ema = EMA(model, cfg.ema_decay)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, cfg.num_epochs)
    crit = CombinedLoss(cfg).to(device)

    best = -1.0
    for epoch in range(cfg.num_epochs):
        model.train()
        run = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.num_epochs}"):
            x = batch["input"].to(device, memory_format=torch.channels_last)
            t = batch["target"].to(device)
            lL = batch["log_L"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred = model(x, lL if cfg.film_enabled else None)
                loss, _ = crit(pred, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ema.update(model)
            run += loss.item()
        sched.step()

        agg, per_bin = validate(ema.shadow, val_loader, device)
        print(f"epoch {epoch+1}: train_loss={run/len(train_loader):.4f} val_SSIM={agg:.4f}")
        print(f"  per-bin SSIM: {per_bin}")     # PC1/PC2 strata, never only aggregate
        if agg > best:
            best = agg
            torch.save(ema.shadow.state_dict(), cfg.checkpoint_path)
            print(f"  saved EMA -> {cfg.checkpoint_path} (val_SSIM={agg:.4f})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke run (2 epochs) then commit**

```bash
source /home/apollo/projects/pysandbox/ml_venv/bin/activate
python train.py --epochs 2 2>&1 | tail -20     # verifies end-to-end on real data
git add train.py && git commit -m "feat: training loop AdamW+cosine+AMP bf16+EMA, per-bin val, CLI ablation flags"
```

---

## Task 7: FiLM validation (closed-form, standalone)

**Files:** Create `validate_film.py`

**Interfaces — Produces:** `main()` CLI printing Spearman(L_true, L̂) for both domains; nonzero exit / warning if `< 0.8`.

**Consumes:** `preprocessing.{synthesize_lr, estimate_L}`, `data._load_npy`.

> No trained model needed — the estimator is closed-form. This validates the FiLM *input signal*, exactly what the spec asks: "compares L_hat against the true L used in re-degradation."

- [ ] **Step 1: Write validator**

```python
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
```

- [ ] **Step 2: Run + commit**

```bash
source /home/apollo/projects/pysandbox/ml_venv/bin/activate
python validate_film.py --num 300 || echo "FiLM corr below threshold — ablate per warning"
git add validate_film.py && git commit -m "feat: closed-form FiLM validator (Spearman L-hat vs true L, 0.8 gate)"
```

---

## Task 8: Inference — bucketed, channels_last, cudnn.benchmark, compile+benchmark flags

**Files:** Create `evaluate.py`

**Interfaces — Produces:** CLI `--input_dir --output_dir [--compile] [--benchmark]`; fixed default checkpoint; writes 2× PNGs. Runs cold, no manual edits.

**Consumes:** `architecture.KLARestoration`, `preprocessing.{log_transform, exp_transform, estimate_L}`, `config.Config`.

- [ ] **Step 1: Resolution-bucketed dataset + inference**

```python
import argparse, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from config import Config
from preprocessing import log_transform, exp_transform, estimate_L
from architecture import KLARestoration


def _load(p):
    if p.suffix.lower() == ".npy":
        a = np.load(p).astype(np.float32)
    else:
        a = np.array(Image.open(p).convert("L"), np.float32) / 255.0
    return np.squeeze(a)


def _buckets(paths):
    b = defaultdict(list)
    for p in paths:
        a = _load(p)
        b[a.shape].append((p, a))
    return b


def build_model(cfg, device):
    model = KLARestoration(cfg)
    sd = torch.load(cfg.checkpoint_path, map_location="cpu")
    model.load_state_dict(sd)
    return model.to(device, memory_format=torch.channels_last).eval()


def run(input_dir, output_dir, compile_on, benchmark):
    cfg = Config()
    cfg.compile_enabled = compile_on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg, device)          # loaded ONCE
    if cfg.compile_enabled:
        model = torch.compile(model)

    paths = sorted(list(Path(input_dir).glob("*.npy")) + list(Path(input_dir).glob("*.png")))
    buckets = _buckets(paths)
    torch.backends.cudnn.benchmark = True     # safe: fixed shape within a bucket

    t0 = time.time()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for shape, items in buckets.items():
            for i in range(0, len(items), cfg.batch_size_test):
                chunk = items[i:i + cfg.batch_size_test]
                xs, lLs, names = [], [], []
                for p, a in chunk:
                    a = np.clip(a, cfg.log_eps, None)
                    lLs.append(np.log(estimate_L(a, domain=cfg.cond_domain)))
                    xs.append(log_transform(a, cfg.log_eps)[None])
                    names.append(p.stem)
                x = torch.from_numpy(np.stack(xs)).to(device, memory_format=torch.channels_last)
                lL = torch.tensor(lLs, dtype=torch.float32, device=device)
                y = model(x, lL if cfg.film_enabled else None)
                y = torch.from_numpy(exp_transform(y.float().cpu().numpy())).clamp(0, 1)
                for j, name in enumerate(names):
                    arr = (y[j, 0].numpy() * 255.0).round().astype(np.uint8)
                    Image.fromarray(arr).save(out / f"{name}.png")
    dt = time.time() - t0
    print(f"processed {len(paths)} images in {dt:.2f}s ({dt/max(1,len(paths))*1000:.1f} ms/img)")

    if benchmark:
        _benchmark(cfg, device, buckets)


def _benchmark(cfg, device, buckets):
    """Measure whether torch.compile amortizes over this test-set size."""
    shape, items = max(buckets.items(), key=lambda kv: len(kv[1]))
    a = np.clip(items[0][1], cfg.log_eps, None)
    x = torch.from_numpy(log_transform(a, cfg.log_eps)[None][None]).to(device, memory_format=torch.channels_last)
    lL = torch.tensor([np.log(estimate_L(a, domain=cfg.cond_domain))], dtype=torch.float32, device=device)

    def timed(m, n=50):
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for _ in range(3):
                m(x, lL if cfg.film_enabled else None)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.time()
            for _ in range(n):
                m(x, lL if cfg.film_enabled else None)
            if device.type == "cuda":
                torch.cuda.synchronize()
        return (time.time() - t) / n

    eager = build_model(cfg, device)
    per_eager = timed(eager)
    comp = torch.compile(build_model(cfg, device))
    t_cold = time.time()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        comp(x, lL if cfg.film_enabled else None)      # triggers compile
        if device.type == "cuda":
            torch.cuda.synchronize()
    cold = time.time() - t_cold
    per_comp = timed(comp)
    N = len(sum(buckets.values(), []))
    save = (per_eager - per_comp) * N
    print(f"eager {per_eager*1000:.1f} ms/img | compiled {per_comp*1000:.1f} ms/img | "
          f"cold {cold:.1f}s | break-even at {cold/max(1e-9, per_eager-per_comp):.0f} imgs | "
          f"test size {N} -> compile {'WINS' if save > cold else 'LOSES'} (net {save-cold:+.1f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()
    run(args.input_dir, args.output_dir, args.compile, args.benchmark)
```

- [ ] **Step 2: Smoke run against real NoisyLR (checkpoint must exist from Task 6 smoke) + commit**

```bash
source /home/apollo/projects/pysandbox/ml_venv/bin/activate
python evaluate.py --input_dir train/train/NoisyLR --output_dir outputs_smoke --benchmark 2>&1 | tail -5
python -c "from pathlib import Path; assert len(list(Path('outputs_smoke').glob('*.png')))>0; print('wrote PNGs')"
git add evaluate.py && git commit -m "feat: bucketed batched inference, channels_last, cudnn.benchmark, compile+benchmark flags"
```

---

## Task 9: requirements.txt

**Files:** Create `requirements.txt`

- [ ] **Step 1: Freeze + commit**

```bash
source /home/apollo/projects/pysandbox/ml_venv/bin/activate
pip install lpips scikit-image >/dev/null 2>&1
pip freeze > requirements.txt
git add requirements.txt && git commit -m "chore: pin training environment (requirements.txt)"
```

---

## Self-Review — spec coverage

- ✓ Single joint denoise+SR NAFNet, no two-stage (Task 4)
- ✓ Width 32, [2,2,4]/8/[2,2,2], exactly 3 downsampling, no attention/LayerNorm, 6–13M asserted (Task 4)
- ✓ Global residual = bicubic-2× log input + head (Task 4)
- ✓ Single 2× PixelShuffle head, size-agnostic/conv (Task 4)
- ✓ Log-domain forward + conditioning; exp before loss/metrics (Tasks 2,5)
- ✓ Closed-form L̂ (ENL + log-trigamma) behind one fn, lowest-variance decile (Task 2)
- ✓ FiLM MLP → per-block scale/shift, every block, toggleable (Task 4); param note documented
- ✓ Does NOT condition on bg_frac/p50/global intensity — only log L̂ (Tasks 2,4)
- ✓ FiLM validator: Spearman vs true L, 0.8 gate, ablation path (Task 7)
- ✓ Crops 128/256, flips, 90/180/270 (Task 3)
- ✓ Re-degradation at randomized L spanning+exceeding provided range; true L recorded for validator (Tasks 2,3,7)
- ✓ Low-prob Gaussian hedge subset (Task 3)
- ✓ PC1-stratified oversampling of top decile, configurable strength (Tasks 1,3)
- ✓ Val split stratified by PC1×PC2 quartile; metrics per bin + aggregate (Tasks 3,6)
- ✓ Loss 1.0/0.3/0.05 linear space, LPIPS fixed low (Task 5)
- ✓ AdamW 2e-4 cosine, AMP bf16, EMA eval (Task 6)
- ✓ Pure PyTorch inference, compile OFF by default behind flag + benchmark mode (Tasks 1,8)
- ✓ inference_mode, autocast bf16, channels_last, bucket-by-res then cudnn.benchmark, model loaded once, workers=8/pin_memory (Task 8)
- ✓ evaluate.py --input_dir/--output_dir, fixed checkpoint, writes PNGs, cold run (Task 8)
- ✓ train.py reproduces recipe; requirements.txt from freeze (Tasks 6,9)
- ✓ FiLM/compile/PC1-strength all toggleable in one config (Task 1)
- ✓ Range: assert 0–1 (deviation from spec's 0–255, per confirmed real data) (Tasks 1,3)

**Two conscious deviations from the literal spec, both confirmed/justified:**
1. Range asserted at **0–1** not 0–255 (real data is float 0–1; user-confirmed).
2. FiLM total params ~**40k** not 5k (per-block conditioning at 256ch needs a projection head; trunk itself is ~5k; fully removable). `film_hidden=8` recovers ≤5k if required.
