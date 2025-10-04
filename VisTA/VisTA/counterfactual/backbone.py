import torch
from model.clip import build_model


def load_clip_rn101(txt_length: int = 40, weights_path: str = 'pretrain/RN101.pt'):
    """Load CLIP RN101 backbone (same as original) for reuse in counterfactual branch.
    Returns the CLIP model with .encode_image and .encode_text.
    """
    clip_jit = torch.jit.load(weights_path, map_location='cpu').eval()
    model = build_model(clip_jit.state_dict(), txt_length).float()
    return model
