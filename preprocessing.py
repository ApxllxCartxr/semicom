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
    Returns LR in 0-1 domain, clipped at a small positive floor (>0 for log).
    NO hard upper clip: multiplicative speckle legitimately exceeds 1.
    """
    if rng is None:
        rng = np.random.default_rng()
    lr_clean = downsample2(hr)
    speckle = rng.gamma(shape=L, scale=1.0 / L, size=lr_clean.shape).astype(np.float32)
    lr = lr_clean * speckle
    return np.clip(lr, eps_clip, None).astype(np.float32)


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
    Estimate Gamma looks L from a LINEAR-domain noisy image (0-1).
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


# ---- Tests (run directly, not via pytest) ----

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
