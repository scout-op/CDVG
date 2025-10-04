import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model
from model.decoder import Reason_Decoder, Mask_Decoder
from model.layers import FPN
from model.qformer import QFormer


# ------------------------ Lovasz-Hinge (binary) helpers ------------------------
def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / (union + 1e-6)
    if jaccard.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

def _flatten_binary_scores(logits: torch.Tensor, labels: torch.Tensor, ignore=None):
    logits = logits.reshape(-1)
    labels = labels.reshape(-1)
    if ignore is None:
        return logits, labels
    valid = labels != ignore
    return logits[valid], labels[valid]

def _lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits: (P,), labels in {0,1}
    signs = labels.float() * 2.0 - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.relu(errors_sorted) @ grad
    return loss

def lovasz_hinge_per_image(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-image Lovasz hinge loss (no mean), logits/labels shape: (B,H,W)."""
    B = logits.shape[0]
    losses = []
    for i in range(B):
        li = logits[i].reshape(-1)
        yi = labels[i].reshape(-1)
        if yi.numel() == 0:
            losses.append(li.new_zeros(()))
        else:
            losses.append(_lovasz_hinge_flat(li, yi))
    return torch.stack(losses, dim=0)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class VisTA(nn.Module):
    def __init__(self):
        super(VisTA, self).__init__()
        # Load CLIP weights with a robust path fallback
        clip_path = os.environ.get('CLIP_WEIGHTS', 'pretrain/RN101.pt')
        if not os.path.exists(clip_path):
            alt = 'RN101.pt'
            if os.path.exists(alt):
                clip_path = alt
        clip_model = torch.jit.load(clip_path, map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(), 40).float()
        self.neck = FPN(in_channels=[512, 1024, 512], out_channels=[256, 512, 1024])

        # Q-Former to refine text state with multi-scale visual features
        self.qformer = QFormer(d_model=512, n_heads=8, num_queries=16, num_layers=2, in_chans=(512, 1024, 512))

        self.ReasonDecoder = Reason_Decoder()

        self.MaskDecoder = Mask_Decoder()

        # Explicit change feature toggle (residual injection, shape-preserving)
        self.enable_explicit_change = True
        self.change1 = nn.Sequential(nn.Conv2d(1024, 1024, 1), nn.BatchNorm2d(1024), nn.ReLU())
        self.change2 = nn.Sequential(nn.Conv2d(2048, 2048, 1), nn.BatchNorm2d(2048), nn.ReLU())
        self.change3 = nn.Sequential(nn.Conv2d(1024, 1024, 1), nn.BatchNorm2d(1024), nn.ReLU())
        self.change_gamma = nn.Parameter(torch.tensor(0.0))  # start as no-op for old checkpoints

        self.down1 = Conv(1024, 512)
        self.down2 = Conv(2048, 1024)
        self.down3 = Conv(1024, 512)

        # Type classifier head (8 types) for low-cost supervision
        # Feature: concat of refined text state (512) and GAP of fused visual (512)
        self.type_head = nn.Sequential(
            nn.Linear(512 + 512, 128),
            nn.ReLU(True),
            nn.Linear(128, 8),
        )

        # Temperature head for uncertainty-driven calibration (outputs scalar per sample)
        self.temp_head = nn.Sequential(
            nn.Linear(512 + 512, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
        )

        # Pixel-wise uncertainty map (temperature) head on fused feature fv
        self.enable_temp_map = True
        self.temp_map_head = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        # Constants for simple consistency regularization
        # Align with dataset label mapping in dataset/DataNew.py
        self.TYPE_NAMES = [
            'change_ratio', 'change_or_not', 'change_to_what', 'increase_or_not',
            'decrease_or_not', 'smallest_change', 'largest_change', 'change_from_what'
        ]
        self.TYPE_CHANGE_OR_NOT = 1
        self.NO_CLASS_INDEX = 17

        # Tunable hyperparameters (can be overridden from train.py via model.module.*)
        self.sample_ratio = 0.7              # sample-level hard mining keep ratio
        self.pixel_hard_ratio = 0.25         # pixel-level OHEM ratio per sample
        self.contrast_weight = 0.05          # weight for contrastive loss
        self.temp_reg_weight = 0.01          # weight for temperature regularization
        self.type_weight = 0.1               # weight for type supervision
        self.consistency_weight = 0.05       # weight for no-change consistency
        self.contrast_tau = 0.07             # temperature for InfoNCE
        self.enable_contrast = True
        self.enable_uncertainty = True
        self.enable_hard_mining = True
        # Text->Prompt points
        self.enable_prompt_points = True
        self.num_prompt_points = 4
        # External prompt override (for HRPG integration). Default off.
        self.use_external_prompts = False
        self.external_prompts = None  # dict with optional keys: 'points': (coords, labels), 'boxes': boxes
        # Lovasz and temperature regularization weights
        self.enable_lovasz = True
        self.lovasz_weight = 0.5
        self.temp_map_reg_weight = 0.005

    def forward(self, img1, img2, word, mask=None, answer_vec=None, type_idx=None):
        vis1 = self.backbone.encode_image(img1)
        vis2 = self.backbone.encode_image(img2)
        word, state = self.backbone.encode_text(word)

        x1 = torch.cat([vis1[0], vis2[0]], dim=1)
        x2 = torch.cat([vis1[1], vis2[1]], dim=1)
        x3 = torch.cat([vis1[2], vis2[2]], dim=1)
        if self.enable_explicit_change:
            d1 = vis2[0] - vis1[0]
            d2 = vis2[1] - vis1[1]
            d3 = vis2[2] - vis1[2]
            r1 = self.change1(torch.cat([d1, d1.abs()], dim=1))
            r2 = self.change2(torch.cat([d2, d2.abs()], dim=1))
            r3 = self.change3(torch.cat([d3, d3.abs()], dim=1))
            x1 = x1 + self.change_gamma * r1
            x2 = x2 + self.change_gamma * r2
            x3 = x3 + self.change_gamma * r3

        v1 = self.down1(x1)
        v2 = self.down2(x2)
        v3 = self.down3(x3)

        # refine text state via Q-Former cross-attention over multi-scale features
        state_refined = self.qformer(v1, v2, v3, state)

        fv = self.neck([v1, v2, v3], state_refined)

        mask_temp, ans, src = self.ReasonDecoder(fv, word, state_refined)

        # Build prompt points from text-visual similarity heatmap
        points = None
        boxes = None
        if self.enable_prompt_points:
            # cosine similarity between src (B,C,h,w) and state_refined (B,C)
            B, C, H, W = src.shape
            s_norm = F.normalize(state_refined, dim=1).view(B, C, 1, 1)
            v_norm = F.normalize(src, dim=1)
            sim = (v_norm * s_norm).sum(dim=1)  # (B,H,W)

            # select top-k per sample
            k = max(1, int(self.num_prompt_points))
            sim_flat = sim.view(B, -1)
            topk = torch.topk(sim_flat, k=k, dim=1).indices  # (B,k)
            ys = (topk // W).float()
            xs = (topk % W).float()

            # map to input image coordinate system expected by PromptEncoder (default 512x512)
            try:
                in_h, in_w = self.MaskDecoder.prompt_decoder.input_image_size
            except Exception:
                in_h, in_w = 512, 512
            stride_x = float(in_w) / float(W)
            stride_y = float(in_h) / float(H)
            # center-of-cell coordinates: (idx+0.5)*stride - 0.5 (to counter PE shift in encoder)
            x_img = xs * stride_x + (stride_x * 0.5 - 0.5)
            y_img = ys * stride_y + (stride_y * 0.5 - 0.5)
            coords = torch.stack([x_img, y_img], dim=-1)  # (B,k,2)
            labels = torch.ones((B, k), device=coords.device, dtype=torch.long)  # positive points
            points = (coords, labels)

        # Override by external prompts if enabled (HRPG)
        if self.use_external_prompts and self.external_prompts is not None:
            ext_pts = self.external_prompts.get('points', None)
            ext_boxes = self.external_prompts.get('boxes', None)
            if ext_pts is not None:
                # Expect a tuple (coords, labels) already in PromptEncoder space and device
                points = ext_pts
            if ext_boxes is not None:
                boxes = ext_boxes

        pred = self.MaskDecoder(vis=fv, points=points, boxes=boxes, masks=mask_temp)

        if self.training:
            # resize mask
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:], mode='nearest').detach()
            # Classification loss (fix: use integer class index target)
            target_idx = answer_vec.argmax(dim=1).long()
            ce_per = F.cross_entropy(ans, target_idx, reduction='none')  # (B,)
            loss1 = ce_per.mean()

            # Boundary-friendly loss + OHEM for segmentation
            logits = pred.float()
            target = mask.float()

            # Uncertainty-driven temperature scaling
            # sample-wise temperature (always computed for logs/reg)
            fv_gap = F.adaptive_avg_pool2d(fv, (1, 1)).flatten(1)
            temp_in = torch.cat([state_refined, fv_gap], dim=1)
            temp = 0.5 + 1.5 * torch.sigmoid(self.temp_head(temp_in))  # (B,1) in [0.5,2.0]
            temp = temp.view(-1, 1, 1, 1)

            # pixel-wise temperature map (preferred when enabled)
            if self.enable_temp_map:
                temp_map_small = self.temp_map_head(fv)  # (B,1,h,w)
                if temp_map_small.shape[-2:] != logits.shape[-2:]:
                    temp_map = F.interpolate(temp_map_small, size=logits.shape[-2:], mode='bilinear', align_corners=True)
                else:
                    temp_map = temp_map_small
                temp_map = 0.5 + 1.5 * torch.sigmoid(temp_map)
                logits_s = logits / temp_map
            else:
                logits_s = logits / temp if self.enable_uncertainty else logits

            bce_per = F.binary_cross_entropy_with_logits(logits_s, target, reduction='none')
            # simple Laplacian edge detector to emphasize boundaries
            with torch.no_grad():
                lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=target.device, dtype=target.dtype).view(1, 1, 3, 3)
                edge = torch.abs(F.conv2d(target, lap, padding=1))
                boundary = (edge > 0).float()
                weight = 1.0 + 3.0 * boundary  # boundary weight factor=3
            bce_per = bce_per * weight

            # Pixel-level OHEM per-sample: select top-k pixels for each sample
            B = bce_per.shape[0]
            hard_ratio = float(self.pixel_hard_ratio)
            per_count = bce_per.shape[2] * bce_per.shape[3]
            k_pix = max(1, int(per_count * hard_ratio))
            b_flat = bce_per.view(B, -1)
            loss_bce_ohem_per = b_flat.topk(k_pix, dim=1).values.mean(dim=1)  # (B,)

            # Dice loss (soft)
            probs = torch.sigmoid(logits_s)
            smooth = 1e-6
            inter = (probs * target).sum(dim=(1, 2, 3))  # (B,)
            den = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
            dice = 1 - ((2 * inter + smooth) / (den + smooth))  # (B,)
            loss_dice_per = dice

            # Lovasz (per-image, using logits after temperature scaling)
            if self.enable_lovasz:
                lovasz_per = lovasz_hinge_per_image(logits_s.squeeze(1), target.squeeze(1))  # (B,)
            else:
                lovasz_per = torch.zeros_like(loss_dice_per)

            # Per-sample segmentation loss
            loss2_per = loss_bce_ohem_per + loss_dice_per + float(self.lovasz_weight) * lovasz_per  # (B,)
            loss2 = loss2_per.mean()

            # Optional: supervised type classification head (8-way)
            loss_type = torch.zeros((), device=logits.device)
            if type_idx is not None:
                # type_idx expected shape: (B,)
                type_logits = self.type_head(torch.cat([state_refined, fv_gap], dim=1))
                loss_type = F.cross_entropy(type_logits, type_idx)

            # Optional: simple answer-mask consistency regularization
            # If type is change_or_not and answer is 'no', penalize mask area
            loss_cons = torch.zeros((), device=logits.device)
            if type_idx is not None:
                is_change_or_not = (type_idx == self.TYPE_CHANGE_OR_NOT)
                is_no = is_change_or_not & (target_idx == self.NO_CLASS_INDEX)
                mask_prob = torch.sigmoid(logits_s)
                area = mask_prob.mean(dim=(1, 2, 3))
                # Only penalize samples labeled no-change
                if is_no.any():
                    loss_cons = (area * is_no.float()).mean()

            # Cross-modal region-text contrastive alignment (InfoNCE)
            # Use GT mask to pool region feature; only for samples with positive area
            loss_ctr = torch.zeros((), device=logits.device)
            pos_mask = None
            if self.enable_contrast:
                with torch.enable_grad():
                    tgt_small = F.interpolate(target, size=src.shape[-2:], mode='nearest')  # (B,1,h,w)
                    pos_mask = (tgt_small.sum(dim=(2, 3)).squeeze(1) > 0)
                    if pos_mask.any() and pos_mask.sum() > 1:
                        src_feat = src  # (B,C,h,w)
                        masked_sum = (src_feat * tgt_small).sum(dim=(2, 3))  # (B,C)
                        pix = tgt_small.sum(dim=(2, 3)) + 1e-6  # (B,1)
                        region_feat = masked_sum / pix  # (B,C)
                        reg = F.normalize(region_feat[pos_mask], dim=1)
                        txt = F.normalize(state_refined[pos_mask], dim=1)
                        sim = reg @ txt.t()  # (M,M)
                        logits_c = sim / float(self.contrast_tau)
                        labels = torch.arange(logits_c.size(0), device=logits.device)
                        loss_ctr = 0.5 * (F.cross_entropy(logits_c, labels) + F.cross_entropy(logits_c.t(), labels))

            # Error-driven hard example mining at sample level
            # Combine classification + segmentation per-sample and keep top-q hardest samples
            sample_total = 0.2 * ce_per + loss2_per  # (B,)
            if self.enable_hard_mining:
                sample_ratio = float(self.sample_ratio)
                k_samp = max(1, int(sample_total.shape[0] * sample_ratio))
                loss_main = sample_total.topk(k_samp).values.mean()
            else:
                k_samp = sample_total.shape[0]
                loss_main = sample_total.mean()

            # Regularize temperature(s) to be near 1.0
            loss_temp = (temp.view(-1) - 1.0).abs().mean()
            if self.enable_temp_map:
                loss_temp = loss_temp + float(self.temp_map_reg_weight) * (temp_map - 1.0).abs().mean()

            # Total loss with small weights for auxiliaries (use tunable weights)
            loss = (
                loss_main
                + float(self.type_weight) * loss_type
                + float(self.consistency_weight) * loss_cons
                + float(self.temp_reg_weight) * loss_temp
                + float(self.contrast_weight) * loss_ctr
            )

            # Build training logs dict (as tensors on device)
            if self.enable_uncertainty:
                temp_mean = temp.view(-1).mean()
                temp_std = temp.view(-1).std()
            else:
                temp_mean = torch.tensor(1.0, device=logits.device)
                temp_std = torch.tensor(0.0, device=logits.device)
            if self.enable_temp_map:
                tmap_mean = temp_map.mean().detach()
            else:
                tmap_mean = torch.tensor(1.0, device=logits.device)
            pos_count = pos_mask.sum().float() if (pos_mask is not None) else torch.zeros((), device=logits.device)
            log_dict = {
                'loss_ctr': loss_ctr.detach(),
                'temp_mean': temp_mean.detach(),
                'temp_std': temp_std.detach(),
                'temp_map_mean': tmap_mean,
                'pos_count': pos_count.detach(),
                'kept_samples': torch.tensor(k_samp, device=logits.device, dtype=torch.float32),
                'batch_size': torch.tensor(ce_per.shape[0], device=logits.device, dtype=torch.float32),
                'loss_temp': loss_temp.detach(),
            }
            return pred.detach(), ans.detach(), mask, loss, loss1, loss2, log_dict
        else:
            # Apply temperature at inference for calibrated probabilities
            if self.enable_temp_map:
                temp_map_small = self.temp_map_head(fv)
                if temp_map_small.shape[-2:] != pred.shape[-2:]:
                    temp_map = F.interpolate(temp_map_small, size=pred.shape[-2:], mode='bilinear', align_corners=True)
                else:
                    temp_map = temp_map_small
                temp_map = 0.5 + 1.5 * torch.sigmoid(temp_map)
                pred = torch.sigmoid(pred / temp_map)
            elif self.enable_uncertainty:
                fv_gap = F.adaptive_avg_pool2d(fv, (1, 1)).flatten(1)
                temp_in = torch.cat([state_refined, fv_gap], dim=1)
                temp = 0.5 + 1.5 * torch.sigmoid(self.temp_head(temp_in))  # (B,1)
                temp = temp.view(-1, 1, 1, 1)
                pred = torch.sigmoid(pred / temp)
            else:
                pred = torch.sigmoid(pred)
            if pred.shape[-2:] != mask.shape[-2:]:
                pred = F.interpolate(pred, size=mask.shape[-2:], mode='bicubic', align_corners=True).squeeze(1)
            return pred.detach(), ans.detach()
