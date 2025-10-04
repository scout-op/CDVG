import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import load_clip_rn101
from .align import DeformAlign
from .generator import RollbackUNet
from .losses import compute_cf_losses, compute_style_losses


class CounterfactualModel(nn.Module):
    """
    Counterfactual (no-change) generator pipeline.
    Inputs:
      - img1: (B,3,H,W) normalized as in DatasetNew (ZeroOneNormalize + ImageNet stats)
      - img2: (B,3,H,W) same normalization
      - q_tokens: (B, L) integer tokens (length=40 by default)
      - mask: (B,1,H,W) binary 1=change (optional, for training losses)
    Outputs (dict):
      - t1_warp, flow, t2_star
      - E_pixel: |img2 - t2_star| mean over channels -> (B,1,H,W)
      - E_feat (optional): CLIP feature-space distance upsampled to HxW
      - losses: {'Lchg','Lunchg','Ltv','Ltotal'} when mask is provided
    """

    def __init__(self,
                 txt_length: int = 40,
                 weights_path: str = 'pretrain/RN101.pt',
                 use_feature_energy: bool = True):
        super().__init__()
        self.clip = load_clip_rn101(txt_length=txt_length, weights_path=weights_path)
        self.align = DeformAlign()
        self.gen = RollbackUNet(cond_dim=512, base=32)
        self.use_feature_energy = use_feature_energy

    @torch.no_grad()
    def _encode_image_features(self, img: torch.Tensor):
        # Returns multi-scale features (v2, v3, v4) from ModifiedResNet
        return self.clip.encode_image(img)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor, q_tokens: torch.Tensor,
                mask: torch.Tensor = None):
        # Text conditioning from CLIP
        _, state = self.clip.encode_text(q_tokens)

        # Align t1 to t2 geometry
        t1_warp, flow = self.align(img1, img2)

        # Generate counterfactual t2*
        t2_star = self.gen(t1_warp, img2, state)

        # Energy maps
        E_pixel = torch.mean(torch.abs(img2 - t2_star), dim=1, keepdim=True)

        feats2_all = None
        feats2s_all = None
        E_feat = None
        if self.use_feature_energy:
            # multi-scale features (v2, v3, v4)
            feats2_all = self._encode_image_features(img2)
            feats2s_all = self._encode_image_features(t2_star)
            f2_last = feats2_all[-1]  # (B,C,h,w)
            f2s_last = feats2s_all[-1]
            ef = torch.mean(torch.abs(f2_last - f2s_last), dim=1, keepdim=True)  # (B,1,h,w)
            E_feat = F.interpolate(ef, size=img2.shape[-2:], mode='bilinear', align_corners=False)

        losses = None
        if mask is not None:
            # base counterfactual losses
            base_losses = compute_cf_losses(t1_warp, img2, t2_star, mask)
            # style/noise consistency losses on unchanged area
            style_losses = compute_style_losses(
                t2=img2,
                t2_star=t2_star,
                mask=mask,
                feats2=list(feats2_all) if feats2_all is not None else None,
                feats2s=list(feats2s_all) if feats2s_all is not None else None,
            )
            # merge and add total
            losses = {**base_losses, **style_losses}
            losses["Ltotal_all"] = base_losses["Ltotal"] + style_losses["Lstyle_img"] + style_losses["Lstyle_feat"]

        return {
            't1_warp': t1_warp,
            'flow': flow,
            't2_star': t2_star,
            'E_pixel': E_pixel,
            'E_feat': E_feat,
            'losses': losses,
        }

    @staticmethod
    def predict_mask(energy: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
        """Binarize energy map to a mask with threshold tau."""
        return (energy >= tau).float()
