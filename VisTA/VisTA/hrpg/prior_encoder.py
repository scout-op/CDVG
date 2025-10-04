from typing import Tuple, Optional, List

import torch

from .prior_types import ChangePrior, BoundingBox


def boxes_to_points(boxes: List[BoundingBox], k_per_box: int = 1) -> torch.Tensor:
    """
    Convert boxes to center points (optionally multiple per box).
    Returns tensor of shape (K, 2) in pixel coords (x, y).
    """
    pts = []
    for b in boxes:
        cx = 0.5 * (b.x1 + b.x2)
        cy = 0.5 * (b.y1 + b.y2)
        for _ in range(max(1, k_per_box)):
            pts.append([cx, cy])
    if not pts:
        return torch.empty(0, 2)
    return torch.tensor(pts, dtype=torch.float32)


def encode_prior_to_prompts(
    prior: ChangePrior,
    prompt_image_size: Tuple[int, int] = (512, 512),
    k_points_per_box: int = 1,
) -> Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
    """
    Convert ChangePrior into PromptEncoder-compatible tuples.
      - points: (coords[B,K,2], labels[B,K]) where labels 1=positive, 0=negative
      - boxes: (B, 2, 2) as [[x1,y1],[x2,y2]] in image coordinates
    For now we support a single image (B=1) and scale coordinates are assumed to
    already be in the prompt image space; adapt scaling before calling if needed.
    """
    B = 1
    points = None
    boxes_t = None
    if prior is None or prior.kind == 'none':
        return points, boxes_t

    if prior.boxes:
        # Boxes tensor: (B, 2, 2)
        b = prior.boxes[0]  # single box for now
        boxes_t = torch.tensor([[[b.x1, b.y1], [b.x2, b.y2]]], dtype=torch.float32)
        # Points tensor: centers (B, K, 2) and labels (B, K)
        pts = boxes_to_points(prior.boxes, k_per_box=k_points_per_box)
        if pts.numel() > 0:
            pts = pts.unsqueeze(0)  # (1,K,2)
            labels = torch.ones((B, pts.shape[1]), dtype=torch.long)
            points = (pts, labels)
    elif prior.kind == 'text' and prior.text:
        # Text-only prior: leave points/boxes None; rely on downstream textual encoding if available
        points, boxes_t = None, None
    # feature prior not handled here (would map to dense_prompt)
    return points, boxes_t
