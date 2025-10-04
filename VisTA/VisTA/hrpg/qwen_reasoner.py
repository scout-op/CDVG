from typing import Dict, Optional, Any, List
import json
import os
import re
import tempfile
from PIL import Image

from .prior_types import ChangePrior, BoundingBox
from .config import HRPGConfig


class QwenVLReasoner:
    """
    Wraps a Qwen-VL (or similar VLM) to produce:
      - a textual answer to the question
      - a change prior (text/boxes/features) to guide segmentation

    Backends:
      - vLLM (preferred for multi-modal inference; optional)
      - Transformers (fallback; text-only by default in this prototype)
    """

    def __init__(self, model_name: str = 'Qwen/Qwen-VL-Chat', device: str = 'cuda', cfg: Optional[HRPGConfig] = None) -> None:
        self.device = device
        self.cfg = cfg or HRPGConfig(device=device, qwen_model_name=model_name)
        self.model_name = self.cfg.qwen_model_name or model_name

        self.backend = 'vllm' if self.cfg.use_vllm else 'hf'
        self.available = False

        # vLLM members
        self._vllm = None
        self._processor = None

        # HF members
        self._tok = None
        self._model = None

        if self.backend == 'vllm':
            self._init_vllm()
        else:
            self._init_transformers()

    # -------------------- Backend initializations --------------------
    def _init_vllm(self) -> None:
        try:
            os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
            from vllm import LLM
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            tp = self.cfg.resolve_vllm_tp()
            self._vllm = LLM(
                model=self.model_name,
                trust_remote_code=True,
                gpu_memory_utilization=float(self.cfg.vllm_gpu_mem),
                enforce_eager=bool(self.cfg.vllm_enforce_eager),
                tensor_parallel_size=int(tp),
                seed=0,
            )
            self.available = True
        except Exception:
            self._vllm = None
            self._processor = None
            self.available = False

    def _init_transformers(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
            self.available = True
        except Exception:
            self._tok = None
            self._model = None
            self.available = False

    # -------------------- Utilities --------------------
    @staticmethod
    def _save_pil_temp(img: Image.Image) -> str:
        os.makedirs(os.path.join(tempfile.gettempdir(), 'hrpg_vlm'), exist_ok=True)
        fd, path = tempfile.mkstemp(suffix='.png', prefix='hrpg_', dir=os.path.join(tempfile.gettempdir(), 'hrpg_vlm'))
        os.close(fd)
        img.save(path)
        return path

    def _build_messages(self, img1: Optional[Image.Image], img2: Optional[Image.Image], question: str) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        # Save images to temp paths for processor
        if isinstance(img1, Image.Image):
            p1 = self._save_pil_temp(img1)
            content.append({"type": "image", "image": p1})
        if isinstance(img2, Image.Image):
            p2 = self._save_pil_temp(img2)
            content.append({"type": "image", "image": p2})
        # Ask for both answer and JSON boxes
        ask = (
            f"{question}\n"
            f"If possible, return up to {int(self.cfg.max_boxes_from_vlm)} bounding boxes of changed regions as JSON in a field named 'boxes', "
            f"with list items like {{'x1':<int>,'y1':<int>,'x2':<int>,'y2':<int>}}."
        )
        content.append({"type": "text", "text": ask})
        return [{"role": "user", "content": content}]

    def _prepare_inputs_for_vllm(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        # qwen_vl_utils >= 0.0.14
        from qwen_vl_utils import process_vision_info
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self._processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        return {
            'prompt': text,
            'multi_modal_data': mm_data,
            'mm_processor_kwargs': video_kwargs,
        }

    @staticmethod
    def _parse_boxes_from_text(txt: str) -> List[BoundingBox]:
        # Try to find JSON-like list with key 'boxes'
        try:
            # Find the first JSON object or list
            m = re.search(r"\{.*\}|\[.*\]", txt, re.S)
            if not m:
                return []
            blob = m.group(0)
            data = json.loads(blob.replace("'", '"'))
            if isinstance(data, dict) and 'boxes' in data:
                arr = data['boxes']
            elif isinstance(data, list):
                arr = data
            else:
                return []
            boxes: List[BoundingBox] = []
            for it in arr:
                x1 = float(it.get('x1', 0)); y1 = float(it.get('y1', 0))
                x2 = float(it.get('x2', 0)); y2 = float(it.get('y2', 0))
                boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, score=float(it.get('score', 1.0)), label=str(it.get('label', ''))))
            return boxes
        except Exception:
            return []

    # -------------------- Public API --------------------
    def predict(self, img1, img2, question: str) -> Dict:
        """
        Inputs can be PIL images or tensors.
        Returns a dict with keys: 'answer_text': str, 'prior': ChangePrior
        """
        if not self.available:
            return {'answer_text': '', 'prior': ChangePrior(kind='none')}

        if self.backend == 'vllm' and self._vllm is not None and self._processor is not None:
            from vllm import SamplingParams
            messages = self._build_messages(img1, img2, question)
            inputs = [self._prepare_inputs_for_vllm(messages)]
            params = SamplingParams(
                temperature=float(self.cfg.vllm_temperature),
                max_tokens=int(self.cfg.vllm_max_tokens),
                top_k=int(self.cfg.vllm_top_k),
                stop_token_ids=[],
            )
            outs = self._vllm.generate(inputs, sampling_params=params)
            txt = outs[0].outputs[0].text if outs and outs[0].outputs else ''
            boxes = self._parse_boxes_from_text(txt)
            prior = ChangePrior(kind='boxes' if boxes else 'text', boxes=boxes or None, text=txt if not boxes else None)
            return {'answer_text': txt, 'prior': prior}

        # Transformers fallback: text-only answer (prototype)
        try:
            prompt = f"[Images: T1,T2]\nQuestion: {question}\nAnswer succinctly."
            out_ids = self._model.generate(**self._tok(prompt, return_tensors='pt').to(self._model.device), max_new_tokens=64)
            answer = self._tok.decode(out_ids[0], skip_special_tokens=True)
        except Exception:
            answer = ''
        prior = ChangePrior(kind='text', text=answer or 'Focus on newly appeared man-made structures if any.')
        return {'answer_text': answer, 'prior': prior}

    @staticmethod
    def infer_boxes_from_text(text: str) -> Optional[list]:
        return None
