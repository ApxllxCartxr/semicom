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
    # fp32 throughout: under bf16 the E[x^2]-E[x]^2 variance below loses all
    # significant digits on flat patches, driving the denominator to ~0 and
    # emitting NaN gradients even when the forward loss stays finite.
    x = x.float().clamp(0, 1); y = y.float().clamp(0, 1)
    w = _gaussian_window(ws, device=x.device).to(x.dtype)
    pad = ws // 2
    mu_x = F.conv2d(x, w, padding=pad); mu_y = F.conv2d(y, w, padding=pad)
    mx2, my2, mxy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sx = (F.conv2d(x * x, w, padding=pad) - mx2).clamp_min(0)
    sy = (F.conv2d(y * y, w, padding=pad) - my2).clamp_min(0)
    sxy = F.conv2d(x * y, w, padding=pad) - mxy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    # Floor added back: clamp_min(0) alone still lets the denominator sink to
    # c1*c2 (~9e-8) whenever both patches are near-black (common in the log
    # domain's dark tail). That's finite but produces gradient spikes large
    # enough to survive norm-clipping as a dominant direction and destabilize
    # training over a few steps -- the mechanism behind the run-2 blowup.
    denom = (mx2 + my2 + c1) * (sx + sy + c2)
    s = ((2 * mxy + c1) * (2 * sxy + c2)) / denom.clamp_min(1e-4)
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
        # Clamp to the model's declared output range [0,1] BEFORE Charbonnier.
        # Without this the linear value is unbounded (exp(30)~1e13): a single
        # large log prediction sends the loss non-finite, NaN gradients slip
        # past grad-clip, and the run diverges. SSIM/LPIPS already clamp.
        pred = _exp(pred_log).clamp(0, 1)
        target = _exp(target_log).clamp(0, 1)
        char = torch.sqrt((pred - target) ** 2 + self.eps ** 2).mean()
        s = 1.0 - ssim(pred, target)
        total = self.cfg.charbonnier_weight * char + self.cfg.ssim_weight * s
        parts = {"charbonnier": char.detach(), "ssim": s.detach()}
        # Linear-domain Charbonnier scales its gradient with pixel value, so the
        # dark tail (GT 1st pct ~0.02) barely trains. A log-domain term restores
        # relative-error weighting there.
        if self.cfg.log_char_weight > 0:
            lc = torch.sqrt((pred_log.float() - target_log.float()) ** 2
                            + self.eps ** 2).mean()
            total = total + self.cfg.log_char_weight * lc
            parts["log_char"] = lc.detach()
        if self.lpips_fn is not None:
            p3 = (pred.clamp(0, 1) * 2 - 1).repeat(1, 3, 1, 1)
            t3 = (target.clamp(0, 1) * 2 - 1).repeat(1, 3, 1, 1)
            lp = self.lpips_fn(p3, t3).mean()
            total = total + self.cfg.lpips_weight * lp
            parts["lpips"] = lp.detach()
        return total, parts


# ---- Tests (run directly, not via pytest) ----

def test_loss_zero_on_identical():
    from config import Config
    cfg = Config(); cfg.lpips_weight = 0.0     # skip lpips download in unit test
    crit = CombinedLoss(cfg)
    x = torch.rand(2, 1, 64, 64).clamp(1e-3, 1).log()
    total, parts = crit(x, x)
    # Charbonnier floor on identical inputs is exactly eps (sqrt(0 + eps^2)); SSIM->0.
    assert parts["charbonnier"] <= cfg.charbonnier_eps + 1e-6, parts["charbonnier"]
    assert parts["ssim"] < 1e-3, parts["ssim"]
