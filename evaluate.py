import argparse, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from config import Config, apply_arch
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
    ck = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "arch" in ck:
        apply_arch(cfg, ck["arch"])     # rebuild the exact trained architecture
        state = ck["ema"]
    else:
        state = ck                       # backward-compat: raw state_dict
    model = KLARestoration(cfg)
    model.load_state_dict(state)
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
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
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
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
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
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
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
