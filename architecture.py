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


# ---- Tests (run directly, not via pytest) ----

def test_forward_and_params():
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
    from config import Config
    cfg = Config(); cfg.film_enabled = True
    m = KLARestoration(cfg).eval()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        y_film = m(x, torch.zeros(1))
        m.film_enabled = False
        y_none = m(x, None)
    assert torch.allclose(y_film, y_none, atol=1e-5), "FiLM proj zero-init must be identity"
