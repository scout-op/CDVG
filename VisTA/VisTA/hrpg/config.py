from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class HRPGConfig:
    device: str = 'cuda'
    use_qwen_vl: bool = True
    qwen_model_name: str = 'Qwen/Qwen-VL-Chat'  # override if needed
    max_boxes_from_vlm: int = 5
    points_per_box: int = 1
    prompt_image_size: Tuple[int, int] = (512, 512)
    # Feature backbone for segmentation (reuse VisTA building blocks for now)
    clip_weights_env: str = 'CLIP_WEIGHTS'
    clip_default_path: str = 'pretrain/RN101.pt'
    clip_alt_path: str = 'RN101.pt'
    # Decoder behavior
    multimask_output: bool = False
    # Fallbacks when VLM is not present
    fallback_generate_boxes: bool = False
    fallback_num_boxes: int = 0
    
    # Backend selection for VLM
    use_vllm: bool = False
    vllm_gpu_mem: float = 0.70
    vllm_tensor_parallel: Optional[int] = None  # None -> use torch.cuda.device_count()
    vllm_enforce_eager: bool = False
    vllm_max_tokens: int = 512
    vllm_temperature: float = 0.0
    vllm_top_k: int = -1
    
    # Future: support training and distillation
    enable_training_hooks: bool = False
    
    def resolve_clip_path(self) -> str:
        import os
        p = os.environ.get(self.clip_weights_env, self.clip_default_path)
        if not os.path.exists(p) and os.path.exists(self.clip_alt_path):
            p = self.clip_alt_path
        return p

    def resolve_vllm_tp(self) -> int:
        import torch
        return self.vllm_tensor_parallel or max(1, torch.cuda.device_count())
