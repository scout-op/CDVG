# -*- coding: utf-8 -*-
import argparse
import os
import torch
from typing import List, Dict, Any

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# Windows friendliness
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')


def prepare_inputs_for_vllm(messages: List[Dict[str, Any]], processor: AutoProcessor) -> Dict[str, Any]:
    """Prepare chat template, multi-modal blobs, and processor kwargs for vLLM.

    messages: a list with role/content entries (Qwen-VL chat format)
    processor: transformers AutoProcessor for Qwen-VL
    """
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+ required
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    print(f"video_kwargs: {video_kwargs}")

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


def build_messages(image: str = None, video: str = None, question: str = None) -> List[Dict[str, Any]]:
    """Helper to build a single-turn message following Qwen-VL chat format."""
    content: List[Dict[str, Any]] = []
    if image:
        content.append({"type": "image", "image": image})
    if video:
        content.append({"type": "video", "video": video})
    if question:
        content.append({"type": "text", "text": question})
    if not content:
        # default demo (image OCR)
        content = [
            {
                "type": "image",
                "image": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png",
            },
            {"type": "text", "text": "Read all the text in the image."},
        ]
    return [{"role": "user", "content": content}]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-VL-30B-A3B-Instruct-FP8', help='Checkpoint or model name')
    parser.add_argument('--image', type=str, default=None, help='Image URL or local path')
    parser.add_argument('--video', type=str, default=None, help='Video URL or local path')
    parser.add_argument('--question', type=str, default=None, help='User question')
    parser.add_argument('--gpu_mem', type=float, default=0.70, help='vLLM gpu_memory_utilization')
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.0)
    args = parser.parse_args()

    # Build messages
    messages = build_messages(image=args.image, video=args.video, question=args.question)

    # Load processor
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    inputs = [prepare_inputs_for_vllm(messages, processor)]

    # Create vLLM engine
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem,
        enforce_eager=False,
        tensor_parallel_size=torch.cuda.device_count(),
        seed=0,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_k=-1,
        stop_token_ids=[],
    )

    for i, input_ in enumerate(inputs):
        print('\n' + '=' * 40)
        print(f"Inputs[{i}]: prompt={input_['prompt']!r}")
    print('\n' + '>' * 40)

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print('\n' + '=' * 40)
        print(f"Generated text: {generated_text!r}")
