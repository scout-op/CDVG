import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_l1(x: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """L1 over region m (broadcastable). m in {0,1}."""
    if m is None:
        return F.l1_loss(x, y)
    num = torch.sum(m)
    if num.item() == 0:
        # fallback to global
        return F.l1_loss(x, y)
    return torch.sum(torch.abs(x - y) * m) / (num * x.shape[1])


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    """Total variation (anisotropic)."""
    loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return loss


def compute_cf_losses(t1_warp: torch.Tensor,
                      t2: torch.Tensor,
                      t2_star: torch.Tensor,
                      mask: torch.Tensor = None,
                      w_chg: float = 1.0,
                      w_unchg: float = 1.0,
                      w_tv: float = 0.1):
    """
    Counterfactual training losses.
    Args:
        t1_warp, t2, t2_star: (B,3,H,W) in normalized space
        mask: (B,1,H,W) binary 1=change, 0=no-change (optional)
    Returns:
        dict with Lchg, Lunchg, Ltv, Ltotal
    """
    if mask is None:
        # assume unknown mask: use whole image, but keep separate terms for reporting
        m = torch.ones_like(t2[:, :1])
    else:
        m = mask
    um = 1.0 - m

    Lchg = masked_l1(t2_star, t1_warp, m) * w_chg
    Lunchg = masked_l1(t2_star, t2, um) * w_unchg
    Ltv = tv_loss(t2_star) * w_tv
    Ltotal = Lchg + Lunchg + Ltv
    return {"Lchg": Lchg, "Lunchg": Lunchg, "Ltv": Ltv, "Ltotal": Ltotal}


# -----------------------
# Style / noise consistency on unchanged region
# -----------------------

def _region_stats(x: torch.Tensor, m: torch.Tensor):
    """Compute per-channel mean/std over masked region. x:(B,C,H,W), m:(B,1,H,W)."""
    B, C, H, W = x.shape
    m = m.float()
    if m.shape[-2:] != x.shape[-2:]:
        m = F.interpolate(m, size=x.shape[-2:], mode='nearest')
    m = m.expand(-1, C, -1, -1)
    denom = m.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    mean = (x * m).sum(dim=(2, 3), keepdim=True) / denom
    var = ((x - mean) ** 2 * m).sum(dim=(2, 3), keepdim=True) / denom
    std = (var + 1e-6).sqrt()
    return mean, std


def style_stats_loss(x: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Match channel-wise mean/std on region m: E[|mu_x-mu_y| + |sigma_x-sigma_y|]."""
    mx, sx = _region_stats(x, m)
    my, sy = _region_stats(y, m)
    l_mu = torch.mean(torch.abs(mx - my))
    l_std = torch.mean(torch.abs(sx - sy))
    return l_mu + l_std


def compute_style_losses(
    t2: torch.Tensor,
    t2_star: torch.Tensor,
    mask: torch.Tensor,
    feats2: list = None,
    feats2s: list = None,
    w_img: float = 0.2,
    w_feat: float = 0.1,
):
    """
    Style/noise consistency over unchanged area (1-mask):
      - Image level: match channel mean/std between t2* and t2
      - Feature level: average style loss over provided feature maps
    Returns a dict with weighted losses.
    """
    um = 1.0 - mask.float()
    out = {}
    # image-level
    Ls_img = style_stats_loss(t2_star, t2, um) * w_img
    out["Lstyle_img"] = Ls_img
    # feature-level (optional)
    Ls_feat = torch.tensor(0.0, device=t2.device, dtype=t2.dtype)
    cnt = 0
    if feats2 is not None and feats2s is not None:
        for f_gt, f_pred in zip(feats2, feats2s):
            Ls_feat = Ls_feat + style_stats_loss(f_pred, f_gt, um)
            cnt += 1
        if cnt > 0:
            Ls_feat = (Ls_feat / cnt) * w_feat
    out["Lstyle_feat"] = Ls_feat
    return out
