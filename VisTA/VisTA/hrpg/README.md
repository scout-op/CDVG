# Hierarchical Reasoning & Prompt-Guided Segmentation (HRPG)

This folder houses the prototype of a decoupled pipeline:
- Step 1: A VLM (e.g., Qwen-VL) performs global understanding and high-level reasoning to produce textual answers and a change prior (text/boxes/features).
- Step 2: A Prior Encoder converts the change prior into prompts (points/boxes/text embeddings) consumable by a lightweight segmentation decoder.
- Step 3: A lightweight decoder (SAM-style) segments fine pixels guided by prompts over an image feature backbone.

Current prototype favors a low-risk integration path: use the existing CLIP+FPN feature extractor and SAM-like MaskDecoder from the VisTA codebase for Step 3, while preparing hooks for Qwen-VL outputs in Step 1.

## Files
- `config.py`: Shared configuration dataclass.
- `prior_types.py`: Typed containers for priors (text, boxes, features).
- `qwen_reasoner.py`: A wrapper for Qwen-VL reasoning (with graceful fallback when Qwen-VL is unavailable).
- `prior_encoder.py`: Utilities to convert priors into prompt points/boxes for the PromptEncoder.
- `pipeline.py`: Orchestration of the 3 steps. Extracts features, encodes prompts, and runs the SAM-style MaskDecoder to get the final mask.

## Minimal Usage (prototype, inference only)
```python
from hrpg.pipeline import HRPGPipeline
from PIL import Image

pipe = HRPGPipeline(device='cuda')
img1 = Image.open('path/to/T1.png').convert('RGB')
img2 = Image.open('path/to/T2.png').convert('RGB')
question = 'Which regions show newly built structures?'

out = pipe.run(img1, img2, question)
# out = { 'answer_text': str, 'prior': ChangePrior, 'mask': torch.Tensor[B,1,H,W] }
```

By default the pipeline uses:
- VLM reasoner (if available) to produce a textual answer and a prior; otherwise a naive fallback will return an empty prior or a coarse heuristic.
- CLIP RN101 weights from `pretrain/RN101.pt` (or `RN101.pt`) for features, same as VisTA.

## Roadmap
- 2D RoPE in QFormer and attention-weight return (optional integration point for generating priors from attention maps).
- Add multi-scale points, negative points, and bounding box priors.
- Optionally swap the feature backbone with Qwen-VL vision encoder (if exposed) to make Step 3 fully VLM-driven.
- Add training hooks (distillation of priors, weak/strong consistency, etc.).

## Notes
- This prototype is deliberately non-intrusive: it does not modify the existing `model/` code to remain stable. It reuses the same building blocks (CLIP, FPN, MaskDecoder) to demonstrate the prompt-guided segmentation flow.
- When Qwen-VL is available, plug it by setting the config flag or passing a model name to the reasoner.
