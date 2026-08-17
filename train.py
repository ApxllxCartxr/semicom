import argparse, copy, os
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from config import Config, arch_meta, apply_arch
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
def validate(ema_model, val_loader, device, max_val=None):
    ema_model.eval()
    per_bin = {}
    for vi, batch in enumerate(val_loader):
        if max_val is not None and vi >= max_val:
            break
        x = batch["input"].to(device, memory_format=torch.channels_last)
        t = batch["target"].to(device)
        lL = batch["log_L"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            pred = ema_model(x, lL)
        pred_l = _exp(pred.float()).clamp(0, 1)
        tgt_l = _exp(t.float()).clamp(0, 1)
        for b in range(x.size(0)):
            s = int(batch["stratum"][b])
            val = ssim(pred_l[b:b+1], tgt_l[b:b+1]).item()
            per_bin.setdefault(s, []).append(val)
    agg = np.mean([v for vs in per_bin.values() for v in vs])
    return agg, {k: float(np.mean(v)) for k, v in sorted(per_bin.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no_film", action="store_true", help="ablate FiLM branch")
    ap.add_argument("--pc1_strength", type=float, default=None)
    ap.add_argument("--max_iters", type=int, default=None, help="cap train batches/epoch (smoke)")
    ap.add_argument("--max_val", type=int, default=None, help="cap val items (smoke)")
    ap.add_argument("--no_lpips", action="store_true", help="drop LPIPS term (CPU speed; changes recipe)")
    ap.add_argument("--workers", type=int, default=None, help="override DataLoader num_workers")
    ap.add_argument("--fast", action="store_true",
                    help="fastest CPU config: smaller model (expand=2, ~5M) + no LPIPS. NOT the canonical recipe.")
    ap.add_argument("--resume", action="store_true", help="resume from --state_path if it exists")
    ap.add_argument("--state_path", default="train_state.pt", help="full-state file for resume")
    args = ap.parse_args()

    cfg = Config()
    if args.fast:
        cfg.naf_block_expand = 2      # ~5M params
        cfg.lpips_weight = 0.0        # biggest per-iter CPU saving
    if args.epochs: cfg.num_epochs = args.epochs
    if args.no_film: cfg.film_enabled = False
    if args.pc1_strength is not None: cfg.pc1_oversample_strength = args.pc1_strength
    if args.no_lpips: cfg.lpips_weight = 0.0
    if args.workers is not None: cfg.num_workers = args.workers

    # If resuming, the checkpoint's architecture wins — apply it BEFORE building the model.
    resume_ck = None
    if args.resume and os.path.exists(args.state_path):
        resume_ck = torch.load(args.state_path, map_location="cpu", weights_only=False)
        apply_arch(cfg, resume_ck["arch"])
        print(f"Resuming from {args.state_path} (epoch {resume_ck['epoch']+1}, best {resume_ck['best']:.4f})")

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
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"(expand={cfg.naf_block_expand}, film={cfg.film_enabled}, lpips_w={cfg.lpips_weight})")
    ema = EMA(model, cfg.ema_decay)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=cfg.restart_period_epochs, T_mult=cfg.restart_mult, eta_min=cfg.lr_min)
    warmup = None
    if cfg.warmup_iters > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.05, total_iters=cfg.warmup_iters)
    gstep = 0
    crit = CombinedLoss(cfg).to(device)

    best = -1.0
    start_epoch = 0
    if resume_ck is not None:
        model.load_state_dict(resume_ck["model"])
        ema.shadow.load_state_dict(resume_ck["ema"])
        opt.load_state_dict(resume_ck["opt"])
        sched.load_state_dict(resume_ck["sched"])
        best = resume_ck["best"]
        start_epoch = resume_ck["epoch"] + 1

    def save_state(epoch):
        torch.save({
            "model": model.state_dict(), "ema": ema.shadow.state_dict(),
            "opt": opt.state_dict(), "sched": sched.state_dict(),
            "epoch": epoch, "best": best, "arch": arch_meta(cfg),
        }, args.state_path)

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        run = 0.0
        n_batches = 0
        n_skipped = 0
        for bi, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.num_epochs}")):
            if args.max_iters is not None and bi >= args.max_iters:
                break
            x = batch["input"].to(device, memory_format=torch.channels_last)
            t = batch["target"].to(device)
            lL = batch["log_L"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                pred = model(x, lL if cfg.film_enabled else None)
                loss, _ = crit(pred, t)
            # A non-finite loss would push NaN gradients past grad-clip and
            # permanently poison the weights + EMA. Skip the batch instead.
            if not torch.isfinite(loss):
                n_skipped += 1
                continue
            loss.backward()
            # clip_grad_norm_ returns the PRE-clip total norm. If any grad is
            # non-finite that norm is NaN, and the clip scales EVERY parameter's
            # gradient by NaN -- one bad batch permanently kills the model and
            # the EMA. A finite loss does not imply finite grads, so check here.
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if not torch.isfinite(gnorm):
                opt.zero_grad(set_to_none=True)
                n_skipped += 1
                continue
            n_batches += 1
            opt.step()
            ema.update(model)
            run += loss.item()
            if warmup is not None and gstep < cfg.warmup_iters:
                warmup.step()
            gstep += 1
        sched.step()

        agg, per_bin = validate(ema.shadow, val_loader, device, max_val=args.max_val)
        skip_note = f" (skipped {n_skipped} non-finite)" if n_skipped else ""
        print(f"epoch {epoch+1}: train_loss={run/max(1,n_batches):.4f} val_SSIM={agg:.4f}{skip_note}")
        print(f"  per-bin SSIM: {per_bin}")     # PC1/PC2 strata, never only aggregate
        if agg > best:
            best = agg
            torch.save({"ema": ema.shadow.state_dict(), "arch": arch_meta(cfg)},
                       cfg.checkpoint_path)
            print(f"  saved best EMA -> {cfg.checkpoint_path} (val_SSIM={agg:.4f})")
        save_state(epoch)   # full state every epoch, so --resume can continue


if __name__ == "__main__":
    main()
