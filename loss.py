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
