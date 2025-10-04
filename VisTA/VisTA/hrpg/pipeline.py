from typing import Dict, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from .config import HRPGConfig
from .qwen_reasoner import QwenVLReasoner
from .prior_encoder import encode_prior_to_prompts
from model.clip import build_model
from model.layers import FPN
from model.decoder import Mask_Decoder
from model.VisTA import Conv


class HRPGPipeline:
    """
    Orchestrates:
      1) VLM reasoning to produce answer/prior
      2) Prior encoding to prompts
      3) Lightweight prompt-guided segmentation (CLIP+FPN+SAM-style decoder)

    This is an inference-oriented prototype. It intentionally reuses VisTA
    building blocks (CLIP RN101, FPN, Mask_Decoder) to minimize risk.
    """

    def __init__(self, cfg: Optional[HRPGConfig] = None, device: str = 'cuda') -> None:
        self.cfg = cfg or HRPGConfig(device=device)
        self.device = self.cfg.device

        # VLM reasoner (graceful when not available)
        self.reasoner = QwenVLReasoner(model_name=self.cfg.qwen_model_name, device=self.device, cfg=self.cfg) if self.cfg.use_qwen_vl else None

        # Feature backbone (CLIP RN101) matching VisTA
        clip_path = self.cfg.resolve_clip_path()
        clip_model = torch.jit.load(clip_path, map_location='cpu').eval()
        self.backbone = build_model(clip_model.state_dict(), 40).float().to(self.device)

        # Downsample convs to align channels with FPN expectations
        self.down1 = Conv(1024, 512).to(self.device)
        self.down2 = Conv(2048, 1024).to(self.device)
        self.down3 = Conv(1024, 512).to(self.device)

        # FPN (same as VisTA)
        self.neck = FPN(in_channels=[512, 1024, 512], out_channels=[256, 512, 1024]).to(self.device)

        # Prompt-guided decoder (SAM-like)
        self.mask_decoder = Mask_Decoder().to(self.device)

    @staticmethod
    def _to_tensor(img: Image.Image) -> torch.Tensor:
        # Expect PIL RGB; normalize like CLIP expects (delegated to internal encode_image)
        import torchvision.transforms as T
        tr = T.Compose([T.ToTensor()])
        return tr(img).unsqueeze(0)  # (1,3,H,W)

    def _extract_features(self, img1: Image.Image, img2: Image.Image) -> torch.Tensor:
        x1 = self._to_tensor(img1).to(self.device)
        x2 = self._to_tensor(img2).to(self.device)
        with torch.no_grad():
            vis1 = self.backbone.encode_image(x1)
            vis2 = self.backbone.encode_image(x2)
            v1 = self.down1(torch.cat([vis1[0], vis2[0]], dim=1))
            v2 = self.down2(torch.cat([vis1[1], vis2[1]], dim=1))
            v3 = self.down3(torch.cat([vis1[2], vis2[2]], dim=1))
            # Fuse
            # Note: FPN expects a refined text state in VisTA; here we pass zeros to keep interface
            B = 1
            state_refined = torch.zeros((B, 512), device=self.device, dtype=v1.dtype)
            fv = self.neck([v1, v2, v3], state_refined)
        return fv

    def _scale_prompts(self, points, boxes, source_hw, target_hw):
        """Scale point/box prompts from source (H,W) to target (H,W)."""
        if points is not None:
            coords, labels = points
            sy = float(target_hw[0]) / float(max(1, source_hw[0]))
            sx = float(target_hw[1]) / float(max(1, source_hw[1]))
            coords = coords.clone()
            coords[..., 0] = coords[..., 0] * sx
            coords[..., 1] = coords[..., 1] * sy
            points = (coords, labels)
        if boxes is not None:
            sy = float(target_hw[0]) / float(max(1, source_hw[0]))
            sx = float(target_hw[1]) / float(max(1, source_hw[1]))
            boxes = boxes.clone()
            boxes[..., 0] = boxes[..., 0] * sx
            boxes[..., 1] = boxes[..., 1] * sy
        return points, boxes

    def run(self, img1: Image.Image, img2: Image.Image, question: str) -> Dict:
        # Step 1: Reasoning
        if self.reasoner is not None and self.reasoner.available:
            r = self.reasoner.predict(img1, img2, question)
        else:
            r = {'answer_text': '', 'prior': None}
        prior = r.get('prior', None)

        # Step 2: Encode prior to prompts
        points, boxes = encode_prior_to_prompts(prior, prompt_image_size=self.cfg.prompt_image_size, k_points_per_box=self.cfg.points_per_box) if prior else (None, None)

        # Step 3: Features + prompt-guided decoding
        fv = self._extract_features(img1, img2)
        # Scale prompts to decoder's expected coordinate space
        try:
            in_h, in_w = self.mask_decoder.prompt_decoder.input_image_size
        except Exception:
            in_h, in_w = self.cfg.prompt_image_size
        src_h, src_w = img2.size[1], img2.size[0]  # PIL size: (W,H)
        points, boxes = self._scale_prompts(points, boxes, (src_h, src_w), (in_h, in_w))
        with torch.no_grad():
            pred = self.mask_decoder(vis=fv, points=points, boxes=boxes, masks=None)
            prob = torch.sigmoid(pred)
        return {
            'answer_text': r.get('answer_text', ''),
            'prior': prior,
            'mask': prob  # (1,1,H,W) at decoder output size
        }
