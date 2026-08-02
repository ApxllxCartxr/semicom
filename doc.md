# KLA Joint Despeckle + 2× Super-Resolution — Project Handbook

Everything needed to run, train, evaluate, and extend this codebase. Read this
**before** running anything. Written for someone who has never seen this repo.

---

## 1. What this is

A single neural network that takes a **noisy, low-resolution** microscopy image
and produces a **denoised, 2× upscaled** clean image, in one pass. The noise is
multiplicative **Gamma speckle** (like SAR/ultrasound), not additive Gaussian.

- Input example: `128×128` noisy → Output: `256×256` clean.
- Also supports `256→512` (the network is fully convolutional & size-agnostic).
- Trained on 3200 ground-truth / degraded `.npy` pairs.
- Scored on **SSIM / PSNR / LPIPS**, with **runtime as a tiebreaker**.

The design is settled and implemented; Section 6 explains every decision so you
can extend it without re-litigating choices.

---

## 2. Repository layout

```
semicon/
├── config.py            # ALL knobs live here (single source of truth)
├── preprocessing.py     # log/exp domain, Gamma-speckle synthesis, L̂ estimator
├── data.py              # loading, re-degradation aug, PC1 stratification, splits
├── architecture.py      # NAFNet + FiLM conditioning + PixelShuffle SR head
├── loss.py              # Charbonnier + SSIM + LPIPS (linear space)
├── train.py             # training loop (AdamW/cosine/AMP-bf16/EMA)
├── evaluate.py          # batched inference -> PNGs (the scored deliverable)
├── validate_film.py     # sanity gate: does the FiLM conditioning signal survive?
├── requirements.txt     # frozen environment
├── regime_analysis.py   # EDA: how many noise "regimes" are in the data?
├── conditioning.py      # EDA: which stats survive the noise (drove FiLM design)?
├── doc.md               # this file
├── docs/superpowers/plans/2026-08-01-kla-despeckle-sr.md   # the build plan
├── train/train/GT/      # 3200 ground-truth .npy  (256×256, float32, 0–1)
├── train/train/NoisyLR/ # 3200 degraded .npy      (128×128, float32, ~[-0.1,1.9])
└── Test_NoisyLR/        # held-out noisy inputs (unseen sources) to restore
```

---

## 3. Environment setup

A virtualenv already exists at `/home/apollo/projects/pysandbox/ml_venv` on the
current machine. On a **fresh GPU box**, recreate it:

```bash
python -m venv ml_venv            # Python 3.11–3.14 all fine
source ml_venv/bin/activate
pip install -r requirements.txt   # pins torch 2.13, torchvision, lpips, etc.
```

> **GPU wheels:** `requirements.txt` pins `torch==2.13.0` (built `+cu130` on this
> box). On your GPU machine install the CUDA build that matches your driver from
> https://pytorch.org — e.g. `pip install torch torchvision --index-url
> https://download.pytorch.org/whl/cu124`. Everything else in requirements is
> pure-Python and portable.

First run of anything touching the loss **downloads AlexNet weights (233 MB)**
for LPIPS into `~/.cache/torch/hub`. One-time; needs internet.

Sanity-check the install:

```bash
python -c "import torch; print(torch.__version__, 'cuda', torch.cuda.is_available())"
```

---

## 4. CPU vs GPU — there is ONE code path

`train.py` and `evaluate.py` auto-detect the device:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**You do not switch modes.** On a CUDA box the same command automatically uses:
- the GPU,
- `channels_last` memory format,
- `torch.autocast(bf16)` mixed precision,
- `cudnn.benchmark` (inference, after resolution bucketing),
- EMA weights for evaluation.

The only manual GPU decision is `torch.compile` for inference, which is **off by
default on purpose** (see §7.3) and lives behind `--compile`.

**Autocast is device-aware:** mixed-precision bf16 is enabled **only on CUDA**.
On CPU the code runs in **fp32**, because CPU *emulates* bf16 and it is
**~8× slower** there (measured 17.2 vs 2.0 s/iter at batch 16). You get this
automatically — no flag. This single detail is why CPU epochs are minutes, not
an hour.

| Concern | CPU (this machine) | GPU (your machine) |
|---|---|---|
| Precision | fp32 (auto) | bf16 autocast (auto) |
| Per-iter (batch 16, 64→128) | ~2 s | ≪1 s |
| Epoch (160 iters) | ~5–6 min train | seconds–minutes |
| Full 100-epoch train | ~9–12 h (feasible overnight) | ~1 h or less |
| Recommended `batch_size` | 8–16 | 32–64 (edit `config.py`) |
| Recommended `num_workers` | 4–8 | 8–16 |
| LPIPS in loss | adds cost; `--no_lpips` to drop | negligible |
| Validation each epoch | cap with `--max_val` | full |

> **bf16 vs fp16 on GPU:** autocast uses **bfloat16** on CUDA. Ampere (A100/RTX
> 30xx) and newer support it natively. On older GPUs (e.g. V100/T4) bf16 works
> but may be slow — change the `dtype=torch.bfloat16` lines in
> `train.py`/`evaluate.py` to `torch.float16` and add a `GradScaler` if you see
> instability. On Ampere+ leave it as bf16 (no scaler needed).

---

## 5. How to run

Always `source ml_venv/bin/activate` first. All commands run from `semicon/`.

### 5.1 Train (the recipe)

```bash
# GPU: full recipe, exactly as specified
python train.py

# Ablations (single toggle each; see §8)
python train.py --no_film                 # remove FiLM conditioning branch
python train.py --pc1_strength 1.0        # disable PC1 oversampling
python train.py --epochs 100              # override epoch count

# CPU-friendly: cap per-epoch validation so epochs don't stall on 640 full frames
python train.py --max_val 96
# CPU smoke (proves wiring in ~1 min, NOT a trained model):
python train.py --epochs 1 --max_iters 3 --max_val 4

# FASTEST CPU baseline (smaller 5M model + no LPIPS + fp32) — NOT the canonical recipe:
python train.py --fast --epochs 15 --max_val 96 --workers 4 --resume
```

### 5.1.1 Resumable training (survives interruptions / multiple sessions)

Every epoch, `train.py` writes full state (`model`, `EMA`, `optimizer`,
`scheduler`, `epoch`, `best`, and **architecture metadata**) to
`train_state.pt`. To continue an interrupted run, just add `--resume` with the
**same** flags:

```bash
python train.py --fast --epochs 15 --max_val 96 --resume     # continues from last epoch
```

- `--resume` is a no-op if `train_state.pt` doesn't exist (starts fresh), so it's
  safe to always include.
- The checkpoint records the architecture, so a `--fast` (5M) run and a canonical
  (10.88M) run each reload correctly — **`evaluate.py` rebuilds whatever arch was
  trained**, no flags needed. You can therefore keep separate runs by pointing
  `--state_path`/`checkpoint_path` at different files.
- `--fast` = `naf_block_expand=2` (5M) + `lpips_weight=0`. On CPU it's ~26× faster
  per image than the canonical model (measured 80 ms vs 2067 ms/img). Use it for a
  CPU baseline; the **GPU person should NOT use `--fast`** — they want the
  canonical 10.88M + LPIPS recipe (`python train.py`).

Training prints per-epoch `train_loss`, aggregate `val_SSIM`, **and per-bin SSIM**
(PC1×PC2 strata — never hides a regime-specific failure in the average). It saves
the **EMA** weights to `checkpoint_ema.pt` whenever aggregate val SSIM improves.
Stop anytime (Ctrl-C); the best checkpoint is already on disk.

Flags added for tractability/ablation:
- `--fast` — fastest CPU config (5M model + no LPIPS). Not the canonical recipe.
- `--resume` / `--state_path P` — resume full training state from `P` (default `train_state.pt`).
- `--max_iters N` — cap training batches per epoch (smoke only; do not use for a real model).
- `--max_val N` — validate on the first N val items (speed; slightly biases which bins are covered).
- `--no_lpips` — drop the LPIPS term (CPU speed; changes recipe). `--workers N` — DataLoader workers.
- `--no_film` — ablate FiLM. `--pc1_strength F` — set oversampling strength. `--epochs N`.

### 5.2 Evaluate (the scored deliverable)

```bash
python evaluate.py --input_dir Test_NoisyLR --output_dir outputs
```

- Reads a fixed checkpoint path (`config.checkpoint_path = checkpoint_ema.pt`).
- Buckets inputs by resolution into fixed-shape batches, then enables
  `cudnn.benchmark`. Writes one `<stem>.png` (2× size) per input. Runs cold with
  no manual edits.
- `--compile` turns on `torch.compile`; `--benchmark` reports whether compile’s
  cold-start cost amortizes over your test-set size (it prints the break-even
  image count). On a small set, compile **loses** — leave it off.

### 5.3 Validate the FiLM conditioning (do this once)

```bash
python validate_film.py --num 300
```

Synthesizes speckle at **known** L, estimates L̂ with the closed-form estimator,
and reports Spearman correlation. **Exit 0 / “OK” means the FiLM signal is real**
(threshold 0.8; we measured 0.847 log-domain). If it ever drops below 0.8, the
FiLM branch is worthless — train with `--no_film`. No trained model needed; this
tests the input signal, not the network.

### 5.4 Exploratory analysis (context, optional)

```bash
python regime_analysis.py --gt_dir train/train/GT      # how many noise regimes?
python conditioning.py --gt_dir train/train/GT --deg_dir train/train/NoisyLR
```

These drove the design (see §6.4) and are not part of the train/eval loop.

---

## 6. Architecture & the reasoning behind it

### 6.1 The pipeline, end to end
```
noisy LR (linear 0–1)
  └─ log(x+eps) ──────────────────────────────┐  (speckle is multiplicative;
                                               │   log turns it additive)
  ├─ estimate L̂ from input (log-domain) ──► FiLM MLP ──► per-block scale/shift
  │                                               │
  └─ NAFNet U-Net (3 down / bottleneck / 3 up) ◄──┘   (FiLM modulates every block)
        └─ 2× PixelShuffle SR head ──► residual (2×, log)
              + bicubic-2×(log input)  ──────────► output (log)
                    └─ exp() ──► linear ──► loss / metrics / PNG
```

### 6.2 Network (`architecture.py`)
- **NAFNet U-shape, width 32**, encoder blocks `[2,2,4]`, middle `8`, decoder
  `[2,2,2]`. **Exactly 3 downsampling stages** — deeper pooling destroys the thin
  dendrite-like structures.
- **Lightweight NAF block**: pointwise-expand → depthwise 3×3 → **SimpleGate**
  (halves channels, no activation) → simplified channel attention (SCA) →
  pointwise, plus a pointwise FFN. **No self-attention, no LayerNorm.** `beta`/
  `gamma` residual scales are zero-init so an untrained block is identity.
- **Global residual**: the network only learns the *correction* on top of a
  bicubic 2× upsample of the (log) input.
- **SR head**: a single **2× PixelShuffle** stage — fully convolutional, so one
  head serves both `128→256` and `256→512`.
- **Param count 10.88M** (in the 8–12M target). The only knob that moves this is
  `naf_block_expand` (see §7.2); width and block counts are fixed by spec.

### 6.3 Log-domain everything
Gamma speckle is **multiplicative**: `noisy = clean × speckle`. Taking `log`
makes it **additive**, which the network handles far better. The forward pass and
all conditioning stats are in log-domain; we `exp()` back to linear **before**
the loss and metrics, because SSIM/PSNR/LPIPS are defined in linear space.

### 6.4 FiLM speckle-severity conditioning
- The single scalar fed to conditioning is the **Gamma looks parameter L**
  (higher L = less speckle), estimated **in closed form, no learned predictor**:
  tile the input, take the most homogeneous (lowest-variance) decile, and either
  compute `ENL=(mean/std)²` (linear) or invert `Var[log x]≈trigamma(L)` (log).
  Both are behind `preprocessing.estimate_L(img, domain=...)`; log-domain is the
  default and scored higher (0.847 vs 0.798 Spearman).
- `log(L̂)` → a small MLP → **per-block scale/shift** applied to every NAF block.
- **Why only L?** The EDA in `conditioning.py` showed the detail-density stats
  (grad/variance) are *noise-dominated* — they mostly report the noise level, not
  the signal — while NAFNet’s own SCA already recovers intensity/sparsity stats.
  So the only thing worth conditioning on externally is the noise level itself.
- **Crucial correctness point:** the model conditions on **L̂ estimated from the
  input**, identically at train and test time. The *true* L (known only during
  synthetic augmentation) is used **only** by `validate_film.py`. Feeding true L
  into the model would be a train/test leak — it’s deliberately avoided.
- FiLM is fully removable: `config.film_enabled=False` or `--no_film`.

### 6.5 Data pipeline (`data.py`)
- **Trains on the real provided pairs AND re-degradation augmentation.** Each
  sample, with prob `synth_prob` (default 0.5), synthesizes a fresh LR from GT
  (downsample 2× → Gamma speckle at a random L spanning+exceeding the observed
  range); otherwise it uses the real `NoisyLR`. Crops are **2×-aligned** (GT crop
  `c`, LR crop `c/2` at the matching location).
- Flips + 90/180/270 rotations (structures aren’t orientation-dependent). A small
  `gaussian_prob` subset adds mild Gaussian noise as a hedge.
- **PC1-stratified oversampling**: compute detail-density features on GT, take
  PC1, and oversample the top decile (where a compact model struggles). Strength
  = `pc1_oversample_strength`.
- **Validation split stratified by PC1 × PC2 quartile** (16 bins); val uses the
  **real pairs deterministically** and reports metrics **per bin**.

### 6.6 Loss (`loss.py`)
`L = 1.0·Charbonnier + 0.3·(1−SSIM) + 0.05·LPIPS`, all in **linear** space after
`exp()`. **LPIPS weight stays at 0.05 — do not raise it.** This is metrology
data; hallucinated texture that wins LPIPS is *actively wrong*.

### 6.7 Training (`train.py`)
AdamW (lr 2e-4, wd 1e-4), cosine decay, grad-clip 1.0, **AMP bf16**, **EMA**
(decay 0.999) — the EMA copy is what gets evaluated and saved.

---

## 7. Config reference (`config.py`)

Everything is one dataclass. Key fields:

| Field | Default | Meaning |
|---|---|---|
| `film_enabled` | `True` | FiLM conditioning branch (ablation toggle) |
| `pc1_oversample_strength` | `1.5` | top-decile oversample weight (`1.0` = off) |
| `compile_enabled` | `False` | `torch.compile` at inference (also `--compile`) |
| `naf_block_expand` | `4` | NAF block width multiplier → **10.88M params** |
| `synth_prob` | `0.5` | P(use re-degradation vs real pair) per sample |
| `cond_domain` | `"log"` | L̂ estimator domain (`"log"` or `"linear"`) |
| `lpips_weight` | `0.05` | **do not raise** |
| `batch_size` | `16` | raise to 32–64 on GPU |
| `num_epochs` | `100` | full recipe |
| `checkpoint_path` | `checkpoint_ema.pt` | fixed path `evaluate.py` reads |

### 7.1 Ablation matrix (what to flip for the report)
- **FiLM on/off**: `--no_film` (or `film_enabled`). Compare val SSIM per bin.
- **PC1 oversampling**: `--pc1_strength 1.0` vs `1.5`. Watch the high-detail bin.
- **compile**: `evaluate.py --compile --benchmark` for the runtime tiebreaker.

### 7.2 Param budget tuning
If you ever change width/blocks and need to re-hit 8–12M, sweep only
`naf_block_expand`:
```bash
python -c "from config import Config; from architecture import KLARestoration; import dataclasses
for e in (2,3,4,5):
    n=sum(p.numel() for p in KLARestoration(dataclasses.replace(Config(),naf_block_expand=e)).parameters())/1e6
    print(e, round(n,2),'M')"
```
Measured: `2→5.01M, 3→7.79M, 4→10.88M, 5→14.30M`.

### 7.3 Why compile is off by default
`torch.compile` cold-start is 30–90 s and the eval timer includes script startup.
Measured break-even ≈ 378 images; below that, compile **loses**. Turn it on only
if the real test set is large and you confirm a win with `--benchmark`.

---

## 8. Deviations from the original spec (all deliberate)

1. **Data range asserted at 0–1, not 0–255.** The spec assumed 0–255; the actual
   `.npy` files are float32 in 0–1 (degraded dips to ~-0.1 and exceeds 1 from
   speckle). GT confirmed clean `[0,1]`. Gamma clip, metric `data_range`, and PNG
   scaling all use 0–1. (`data.py` asserts GT∈[0,1], degraded∈[-0.2, 5].)
2. **FiLM ≈ 40k params, not the spec’s ~5k.** Per-block conditioning at 256
   channels needs a projection head; the MLP trunk alone is ~5k. Set
   `film_hidden=8` to recover ≤5k if strictly required. Still ≪ the 10.88M budget
   and fully removable.
3. **`--max_iters`/`--max_val` flags** added to make CPU smoke-testing possible.

---

## 9. Known state & next steps

- ✅ All unit tests pass; smoke train + inference verified end-to-end on real data.
- ✅ Param budget 10.88M; FiLM Spearman 0.847 (conditioning justified).
- ⚠️ **`checkpoint_ema.pt` currently on disk is a smoke artifact** (few iterations)
  unless a real run has since overwritten it. A real training run replaces it on
  the first epoch that improves val SSIM. **Do not submit outputs from the smoke
  checkpoint.**
- **To produce the deliverable model:** run `python train.py` (GPU) to completion,
  then `python evaluate.py --input_dir Test_NoisyLR --output_dir outputs`.
- No version control is set up in this directory (by request). A `.gitignore`
  excluding data/venv/checkpoints is drafted if you want to `git init`.

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `AssertionError: degraded range ... out of [-0.2,5.0]` | Inputs aren’t in the expected 0–1 float range. Check your `.npy` source; adjust the assert in `data.py:load_paired_dataset` only if the data legitimately differs. |
| `No degraded match for <stem>` | GT and NoisyLR filenames must share a stem. |
| LPIPS download hangs | No internet for the one-time AlexNet fetch. Pre-place `alex.pth` in `~/.cache/torch/hub/checkpoints/`, or set `lpips_weight=0` (changes the recipe). |
| bf16 slow / unstable on old GPU | Switch autocast `dtype` to `float16` + add `GradScaler` (see §4). |
| Training spends most time in validation (CPU) | Use `--max_val 96`. |
| `torch.compile` makes eval slower | Expected below ~378 images; drop `--compile`. |
| Want a smaller/faster model | Lower `naf_block_expand` (§7.2); re-check the 8–12M target isn’t required for scoring. |
```
