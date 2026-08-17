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
    naf_block_expand: int = 4      # tuned (Task 4 gate): expand=4 -> 10.88M params, inside 8-12M target

    # ---- FiLM conditioning ----
    film_enabled: bool = True                         # ABLATION TOGGLE
    film_hidden: int = 64                             # trunk hidden dim (~5k trunk)

    # ---- Data pipeline ----
    synth_prob: float = 0.5                           # P(use re-degradation) per sample; else real pair
    gaussian_prob: float = 0.1                        # low-prob Gaussian hedge subset
    gaussian_sigma_range: Tuple[float, float] = (0.5 / 255, 2.0 / 255)  # in 0-1 units
    crop_gt_256: int = 192                            # GT crop when GT is 256 -> LR crop 96 (LayerNorm now bounds activations, safe to widen)
    crop_gt_512: int = 256                            # GT crop when GT is 512 -> LR crop 128
    l_range_margin: float = 0.10                      # extend sampled L range 10% beyond observed
    pc1_oversample_strength: float = 1.5              # ABLATION TOGGLE (1.0 = off)
    val_split_ratio: float = 0.2

    # ---- Loss ----
    charbonnier_weight: float = 1.0
    charbonnier_eps: float = 1e-3
    ssim_weight: float = 0.6       # eval metric is SSIM; was 0.3 -> underweighted vs Charbonnier
    log_char_weight: float = 0.2   # relative-error term; trains the dark tail (was 0.5 -> too aggressive, fed SSIM denom instability)
    lpips_weight: float = 0.05                        # DO NOT tune up

    # ---- Optimization ----
    lr: float = 2e-4
    lr_min: float = 1e-6
    warmup_iters: int = 500
    restart_period_epochs: int = 15   # cosine warm-restart cycle length (SGDR)
    restart_mult: int = 1             # cycle length multiplier after each restart
    weight_decay: float = 1e-4
    num_epochs: int = 100
    crops_per_image: int = 4       # sampler draws 4x len(ds)/epoch -> 4x optimizer steps
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    batch_size: int = 8
    num_workers: int = 8
    seed: int = 0

    # ---- Inference ----
    checkpoint_path: str = "checkpoint_ema.pt"
    batch_size_test: int = 4
    compile_enabled: bool = False                     # ABLATION / SPEED TOGGLE


CONFIG = Config()


# ---- Architecture metadata (persisted in checkpoints so eval rebuilds the exact
# model that was trained, even for a --fast CPU baseline with different params) ----

_ARCH_FIELDS = (
    "naf_width", "naf_enc_blocks", "naf_mid_blocks", "naf_dec_blocks",
    "naf_block_expand", "film_enabled", "film_hidden", "log_eps", "sr_scale",
)


def arch_meta(cfg: "Config") -> dict:
    """Snapshot the fields that determine module structure/shape."""
    return {k: getattr(cfg, k) for k in _ARCH_FIELDS}


def apply_arch(cfg: "Config", meta: dict) -> "Config":
    """Overwrite cfg's arch fields from a checkpoint's metadata (in place)."""
    for k in _ARCH_FIELDS:
        if k in meta:
            setattr(cfg, k, meta[k])
    return cfg
