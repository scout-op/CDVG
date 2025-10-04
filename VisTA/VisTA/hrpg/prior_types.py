from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    label: str = ''


@dataclass
class ChangePrior:
    kind: str  # 'text' | 'boxes' | 'features' | 'mixed' | 'none'
    text: Optional[str] = None
    boxes: Optional[List[BoundingBox]] = None
    # Optional feature map prior (e.g., a coarse heatmap provided by the VLM)
    feature: Optional[object] = None  # keep generic to avoid hard deps
