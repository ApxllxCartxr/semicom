# Training Log — KLA Joint Despeckle + 2x SR

## Task

Restore SEM (scanning electron microscope) images from a noisy, low-resolution input to a
clean, 2x super-resolved output — jointly denoising multiplicative speckle noise and doing
2x SR in one pass, conditioned on an estimated noise-level parameter `L`.

- Data: 3200 paired `(GT, NoisyLR)` 256x256 → 128x128 `.npy` image pairs (`train/train/GT`,
  `train/train/NoisyLR`), 80/20 stratified train/val split (16 strata from a 2-component PCA
  over image gradient/variance statistics, so both splits cover the same difficulty spread).
- Baselines to judge any run against: **bicubic upsampling ≈ 0.555 SSIM**, **perfect-denoise
  ceiling ≈ 0.871 SSIM** (the best any model could score given the SR/denoise task itself).

---

## Approach

1. **Preprocessing** — everything is done in the log domain (`log(x + eps)`, `eps = 1e-6`).
   Multiplicative speckle noise becomes additive in log space, and it compresses the huge
   dynamic range of SEM intensities so both bright and near-black regions get proportionate
   gradient signal.
2. **Conditioning** — a per-image noise-level estimate `L` (log domain) is computed from the
   *input* (matching train/test) and fed into the network via FiLM (Feature-wise Linear
   Modulation), so the same backbone adapts its behavior to the local noise regime instead of
   learning one noise level.
3. **Backbone** — a NAFNet-style encoder/middle/decoder with additive skip connections, chosen
   for being a strong, cheap, normalization-light restoration architecture. (See
   [Architecture Bug](#architecture-bug-found-and-fixed) below — "normalization-light" turned
   out to be the wrong call without care.)
4. **SR head** — PixelShuffle 2x upsampler on top of the backbone's LR-resolution features,
   added to a bicubic-upsampled residual base (`residual + bicubic(x)`), so the network only
   has to learn the *correction* to bicubic, not the whole image.
5. **Loss** — combined Charbonnier (linear domain) + SSIM + a log-domain Charbonnier term (to
   fix gradient starvation in dark pixels — see below) + a small LPIPS perceptual term.
6. **Data augmentation** — random 2x-aligned crops, flips/rotations, a 50% chance of
   re-synthesizing the degradation from GT at a freshly sampled noise level `L` (vs. using the
   real provided NoisyLR pair) to widen the effective training distribution, and PC1-based
   oversampling of the "hardest" (highest-detail) decile of images.
7. **Optimization** — AdamW, gradient-norm clipping with an explicit non-finite guard,
   EMA of weights (what's actually checkpointed/evaluated), bf16 autocast, cosine annealing
   with warm restarts (SGDR) in the final recipe.

---

## Architecture (current)

```
KLARestoration
├── NAFNet backbone (naf_width=32)
│   ├── intro: Conv2d(1, 32, 3x3)
│   ├── encoder stage 1: 2x NAFBlock(32ch)  -> down to 64ch
│   ├── encoder stage 2: 2x NAFBlock(64ch)  -> down to 128ch
│   ├── encoder stage 3: 4x NAFBlock(128ch) -> down to 256ch
│   ├── middle:          8x NAFBlock(256ch)
│   ├── decoder stage 1: 2x NAFBlock(128ch)  (+ skip from enc stage 3)
│   ├── decoder stage 2: 2x NAFBlock(64ch)   (+ skip from enc stage 2)
│   └── decoder stage 3: 2x NAFBlock(32ch)   (+ skip from enc stage 1)
├── FiLMGenerator (log_L -> per-channel scale/shift, zero-init = identity at init)
│   MLP: Linear(1,64) -> GELU -> Linear(64,64) -> GELU -> Linear(64, 2*256)
│   injected into every NAFBlock at every stage (max_c=256 covers the deepest stage;
│   shallower blocks slice the first `channels` entries)
└── SR head: Conv2d(32,128,3x3) -> PixelShuffle(2) -> Conv2d(32,1,3x3)
    output = sr_head(backbone_features) + bicubic_upsample(input, 2x)
```

**NAFBlock** (each of the 20 blocks in the backbone):
```
x -> LayerNorm2d -> Conv1x1(expand) -> DWConv3x3 -> SimpleGate -> SCA -> Conv1x1 -> [FiLM] -> +x (residual, beta-gated)
  -> LayerNorm2d -> Conv1x1(expand, FFN) -> SimpleGate -> Conv1x1 -> +x (residual, gamma-gated)
```
`beta`/`gamma` are learned per-channel scalars, zero-initialized (residual contributes nothing
at init). `SimpleGate` = split channels in half, multiply the two halves elementwise. `SCA`
(simplified channel attention) = global-avg-pool -> 1x1 conv, multiplicative gate.

- `naf_block_expand = 4` (channel expansion ratio inside each block)
- **10.89M parameters** total (target band was 8-12M)
- `sr_scale = 2`

### Architecture bug found and fixed

The original implementation of `NAFBlock` had **no normalization layers at all** — it relied
solely on the zero-initialized `beta`/`gamma` residual gates to keep the network stable at
init, unlike the original NAFNet paper (which places a `LayerNorm2d` before both the
token-mixer and the FFN in every block). That's fine while the gates stay near zero, but once
they train away from zero, ~20 stacked *unnormalized* residual blocks let activations grow
without bound on any input with wide dynamic range — which is common here: a single crop can
span from the `log_eps` floor (near-black) to a saturated bright pixel, a ~13.8 log-unit
range.

This was diagnosed by instrumenting every layer with forward hooks on an actual diverging
checkpoint: `backbone.downs.0` hit `max|activation| = 1.6e8` on a real training batch, and
every block downstream produced NaN under bf16 autocast. The same NaN was then reproduced on a
**freshly initialized** model given a synthetic worst-case input, confirming the failure was
architectural, not a symptom of a previously-corrupted run. Adding `LayerNorm2d` (per-pixel,
channel-wise, matching the ConvNeXt/NAFNet convention) before `conv1` and `conv4` in every
block fixed it — verified the same extreme-input probe stopped producing NaN on a fresh model,
then confirmed empirically: the fixed architecture ran two full training passes (40 and 60
epochs) with **zero non-finite batches**, where every prior attempt without it destabilized at
the same epoch (5-6) regardless of loss-weight tuning. Full root-cause writeup in project
memory (`[[semicom-nan-cascade]]`).

This is a **checkpoint-breaking change** — any checkpoint saved before this fix (`run1`) has a
different state dict and cannot be loaded into the current architecture.

### Loss function (current)

```
total = charbonnier_weight * Charbonnier(pred_linear, target_linear)      [1.0]
      + ssim_weight        * (1 - SSIM(pred_linear, target_linear))       [0.6]
      + log_char_weight    * Charbonnier(pred_log, target_log)            [0.2]
      + lpips_weight       * LPIPS(pred, target)                          [0.05]
```
- `pred`/`target` linear values are `exp(log_pred).clamp(0,1)` — clamped *before* the
  Charbonnier/SSIM terms, otherwise a single large log-domain prediction (`exp(30) ≈ 1e13`)
  sends the loss non-finite immediately.
- The log-domain Charbonnier term (`log_char`) was added because the linear-domain Charbonnier
  gradient scales with pixel value, so the dark tail (GT 1st percentile ≈ 0.02) was starved of
  gradient; the log-domain term restores relative-error weighting for dark pixels.
- SSIM is computed in fp32 always (bf16 loses all significant digits in the
  `E[x²] - E[x]²` variance term on flat patches), and its denominator is floored at `1e-4`
  (not just `clamp_min(0)` on the variances) — without the floor, near-black patches (common in
  the log-domain dark tail) drive the denominator toward its true minimum of `c1*c2 ≈ 9e-8`,
  producing gradient spikes large enough to survive global norm-clipping as a dominant
  direction and destabilize training over a handful of steps.
- `torch.nn.utils.clip_grad_norm_` returns a single global norm; if *any* parameter's gradient
  is non-finite, that norm is NaN and the clip multiplies **every** parameter by NaN. The
  training loop checks `torch.isfinite(gnorm)` and skips the optimizer/EMA step on failure
  instead of applying it — cheap insurance kept regardless of the deeper architecture fix.

---

## Training runs

| Run | Epochs | Crop | Batch | LR schedule | Result (val SSIM) | Outcome |
|---|---:|---:|---:|---|---:|---|
| **Run 1** (baseline recipe) | 100 (died @73) | 128 | 16 | cosine (single cycle) | 0.5649 (epoch 72, best before divergence) | Diverged to NaN at epoch 73; every batch non-finite by epoch 88. Checkpoint preserved as `checkpoint_ema_run1_0.5649.pt`. |
| Run 2 | 6 (stopped) | 192 | 16 | cosine | 0.7049 (epoch 6, before stop) | Same instability pattern re-emerging at epoch 5-6 (skip count 10→158→577); stopped to investigate rather than let it fully collapse. |
| Run 3 (probe) | 8 | 192 | 16 | cosine | 0.6367 | SSIM-denominator floor fix (`clamp_min(1e-4)`) tested; contained but did not eliminate the epoch 5-6 skip-count escalation. |
| Run 4 | 6 (stopped) | 192 | 16 | cosine | 0.6140 (epoch 6, before stop) | Same escalation recurring at the same epoch despite the loss fix — signal that the real cause was upstream of the loss, in the architecture. Root-caused to missing normalization (see above). |
| Run 5 | 1 (OOM) | 192 | 16 | cosine | 0.3988 (epoch 1 only) | First run with `LayerNorm2d` added — epoch 1 had **zero non-finite skips**, confirming the architecture fix. OOM'd at epoch 2 start (8GB GPU too tight at this crop/batch). |
| Run 6 | 1 (crashed) | 128 | 12 | cosine | 0.4338 (epoch 1 only) | Crop/batch reduced for GPU memory; crashed at epoch 2 start from **Windows paging-file exhaustion** — `num_workers=8` was spawning too many DataLoader worker processes for this system's 24GB RAM. |
| **Run 7** | **40/40** | 128 | 12 | cosine (single cycle) | **0.6734** | First fully clean full run: zero non-finite batches across all 40 epochs, `num_workers=2`. Converged/plateaued by ~epoch 25 (epochs 25-40 added only +0.002). |
| **Run 8 (current best)** | **60/60** | **192** | 8 | **cosine warm restarts, 15-epoch cycles** | **0.6879** | Wider crops (more spatial context) + SGDR to escape the run-7 plateau. Each restart (epochs 15, 30, 45) gave a real but shrinking bump (+0.008, +0.002, +0.001) — classic SGDR diminishing-returns pattern, confirmed via per-bin SSIM staying frozen across all 16 strata between restarts (genuine convergence, not metric noise). |

### Run 8 final per-bin SSIM (16 PC1/PC2 strata)

```
{0: 0.692, 1: 0.685, 2: 0.769, 4: 0.612, 5: 0.722, 6: 0.722, 7: 0.699,
 8: 0.556, 9: 0.716, 10: 0.728, 11: 0.725, 12: 0.538, 13: 0.704, 14: 0.670, 15: 0.724}
```
Bins 8 (0.556) and 12 (0.538) are the clear bottleneck dragging the aggregate down — likely the
highest-detail/highest-noise strata (the ones `pc1_oversample_strength` already up-weights).
Bin 2 (0.769) is the strongest. The aggregate ceiling is set by these lagging bins, not a
broad failure across the board.

---

## Current best checkpoint

- File: `checkpoint_ema.pt` (43.7MB, EMA weights only + arch metadata)
- **val_SSIM = 0.6879**
- vs. bicubic baseline (0.555): **+0.133**
- vs. perfect-denoise ceiling (0.871): **-0.183** (still headroom)
- Trained with: 192px crops, batch 8, 60 epochs, cosine warm restarts (`T_0=15`), the fixed
  loss weights (`ssim_weight=0.6`, `log_char_weight=0.2`), `LayerNorm2d`-fixed architecture.
- `checkpoint_ema_run1_0.5649.pt` is kept as the pre-fix reference point.

---

## Infrastructure notes (this machine)

- GPU: NVIDIA RTX 5050 Laptop, **8GB VRAM** — crop size and batch size are jointly memory-
  constrained; 192px crops required dropping batch size 16→8 vs. the 128px runs.
- System RAM: **24GB total** — `num_workers` must stay ≤2. At `num_workers=8`, DataLoader
  worker processes (each re-importing torch/CUDA) exhausted the Windows paging file and
  crashed training at the start of the second epoch.
- Training speed at the final recipe (192px crop, batch 8, 2 workers): roughly 5-8 it/s once
  warmed up.

---

## Where to push next (not yet attempted)

The run-8 plateau (per-bin SSIM frozen between restarts) means more epochs on the *same*
recipe won't move the needle further. Untried levers, roughly in order of expected impact:
1. **Targeted work on the low strata** (bins 8, 12) — these are the aggregate bottleneck;
   understand what distinguishes them (likely highest-noise/highest-detail images) and either
   oversample them further or give the loss more weight there.
2. **More model capacity** (`naf_mid_blocks` 8→10-12, or `naf_width` up) — now that
   `LayerNorm2d` bounds activations, a deeper/wider model should train safely; the current
   10.9M params was tuned to a param-count *target*, not a capacity ceiling.
3. **More/better data** — `synth_prob` and the re-degradation sampling range control how much
   the model sees synthetic vs. real noise; worth auditing whether the synthetic degradation
   model matches the real `NoisyLR` distribution closely enough, especially in the low-scoring
   strata.
4. **Longer SGDR schedule** (more/longer restart cycles) if capacity/data changes don't exhaust
   the ceiling on their own.
