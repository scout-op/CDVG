import torch
import torch.nn as nn
import torch.nn.functional as F


def warp_with_flow(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp image by dense flow using grid_sample.
    Args:
        img: (B, C, H, W)
        flow: (B, 2, H, W) flow in pixels (dx, dy)
    Returns:
        warped: (B, C, H, W)
    """
    B, C, H, W = img.shape
    # build base grid in [-1, 1]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=img.device, dtype=img.dtype),
        torch.linspace(-1, 1, W, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

    # normalize flow to [-1, 1]
    norm_flow_x = flow[:, 0] / ((W - 1.0) / 2.0)
    norm_flow_y = flow[:, 1] / ((H - 1.0) / 2.0)
    grid = base_grid + torch.stack([norm_flow_x, norm_flow_y], dim=-1)

    warped = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return warped


class DeformAlign(nn.Module):
    """Lightweight flow estimator for t1->t2 alignment.
    Input: concat(t1, t2) in [0,1], (B, 6, H, W)
    Output: flow (B, 2, H, W) in pixel units and warped t1.
    """

    def __init__(self, channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, channels, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(channels, channels * 2, 3, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(channels, 32, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(32, 2, 3, 1, 1),
        )

    def forward(self, t1: torch.Tensor, t2: torch.Tensor):
        x = torch.cat([t1, t2], dim=1)
        flow = self.net(x)
        t1_warp = warp_with_flow(t1, flow)
        return t1_warp, flow
