import os
import json
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI  # type: ignore
except Exception as e:  # pragma: no cover
    OpenAI = None


class LLMClient:
    """
    OpenAI SDK-based Chat Completions client supporting OpenAI/DeepSeek/local services.
    - Online mode (calls HTTP API via openai SDK). For offline/mock, set mode='mock'.
    - Reads configuration from env if not provided:
        For OpenAI: OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL
        For DeepSeek: DEEPSEEK_API_KEY (auto-detects base_url)
    """
    
    # ============ 在此处填写你的 API Key ============
    DEFAULT_DEEPSEEK_KEY = ""  # 填写你的 DeepSeek API Key，例如: "sk-xxxxx"
    DEFAULT_OPENAI_KEY = ""    # 填写你的 OpenAI API Key（可选）
    # ===============================================

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
        mode: str = "online",
        provider: str = "auto",  # "auto", "openai", "deepseek"
    ) -> None:
        self.mode = mode
        self.timeout = timeout
        
        # Auto-detect provider if not specified
        if provider == "auto":
            if os.getenv("DEEPSEEK_API_KEY"):
                provider = "deepseek"
            else:
                provider = "openai"
        
        # Configure based on provider
        if provider == "deepseek":
            self.base_url = base_url or "https://api.deepseek.com"
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or self.DEFAULT_DEEPSEEK_KEY
            self.model = model or "deepseek-chat"
        else:  # openai or custom
            self.base_url = base_url or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY") or self.DEFAULT_OPENAI_KEY
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        self.provider = provider
        
        if self.mode == "online":
            if OpenAI is None:
                raise RuntimeError("openai package is required for online LLM calls. Please `pip install openai`.")
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        else:
            self.client = None

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        if self.mode != "online":
            # mock deterministic template for offline debugging
            return '{"ops": [{"op":"SEG_CHANGE","args":{"source":"E_feat","tau":0.1},"out":"m"},{"op":"AREA","args":{"mask":"m"},"out":"a"},{"op":"THRESHOLD_ANSWER","args":{"metric":"a","threshold":0.001,"pos":"yes","neg":"no"},"out":"answer"}]}'

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        return response.choices[0].message.content
