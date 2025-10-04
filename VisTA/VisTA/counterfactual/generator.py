import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    def __init__(self, in_channels: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, in_channels)
        self.beta = nn.Linear(cond_dim, in_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W); cond: (B, D)
        g = self.gamma(cond).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(cond).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + g) + b


class FilmConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.film = FiLM(out_ch, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.relu(self.bn2(self.conv2(x)), True)
        x = self.film(x, cond)
        return x


class RollbackUNet(nn.Module):
    """U-Net style generator to produce counterfactual t2* from inputs.
    Inputs (concatenated): [t1_warp, t2, diff, abs_diff] => (B, 12, H, W)
    Condition: text state vector (B, D), applied via FiLM.
    Output: residual image r in [-1,1] added to t2 -> t2*.
    """

    def __init__(self, cond_dim: int = 512, base: int = 32):
        super().__init__()
        self.enc1 = FilmConvBlock(12, base, cond_dim)
        self.down1 = nn.Conv2d(base, base * 2, 4, 2, 1)
        self.enc2 = FilmConvBlock(base * 2, base * 2, cond_dim)
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, 2, 1)
        self.enc3 = FilmConvBlock(base * 4, base * 4, cond_dim)
        self.down3 = nn.Conv2d(base * 4, base * 8, 4, 2, 1)
        self.enc4 = FilmConvBlock(base * 8, base * 8, cond_dim)

        self.up1 = nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1)
        self.dec1 = FilmConvBlock(base * 8, base * 4, cond_dim)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1)
        self.dec2 = FilmConvBlock(base * 4, base * 2, cond_dim)
        self.up3 = nn.ConvTranspose2d(base * 2, base, 4, 2, 1)
        self.dec3 = FilmConvBlock(base * 2, base, cond_dim)

        self.out_conv = nn.Conv2d(base, 3, 1)

    def forward(self, t1_warp: torch.Tensor, t2: torch.Tensor, cond: torch.Tensor):
        diff = t2 - t1_warp
        adiff = torch.abs(diff)
        x = torch.cat([t1_warp, t2, diff, adiff], dim=1)

        e1 = self.enc1(x, cond)
        e2 = self.enc2(F.relu(self.down1(e1), True), cond)
        e3 = self.enc3(F.relu(self.down2(e2), True), cond)
        e4 = self.enc4(F.relu(self.down3(e3), True), cond)

        d1 = self.dec1(torch.cat([F.relu(self.up1(e4), True), e3], dim=1), cond)
        d2 = self.dec2(torch.cat([F.relu(self.up2(d1), True), e2], dim=1), cond)
        d3 = self.dec3(torch.cat([F.relu(self.up3(d2), True), e1], dim=1), cond)

        res = torch.tanh(self.out_conv(d3))  # [-1,1]
        t2_star = torch.clamp(t2 + 0.3 * res, min=-5.0, max=5.0)
        return t2_star
