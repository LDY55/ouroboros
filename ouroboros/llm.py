"""
Ouroboros — LLM client.

Supports multiple LLM providers with dynamic routing.
Contract: chat(), default_model(), available_models(), add_usage().
"""

from __future__ import annotations

import logging
import os
import time
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Primary engine
DEFAULT_MODEL = "gemini/gemini-1.5-flash"
DEFAULT_LIGHT_MODEL = DEFAULT_MODEL


def normalize_reasoning_effort(value: str, default: str = "medium") -> str:
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def reasoning_rank(value: str) -> int:
    order = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5}
    return int(order.get(str(value or "").strip().lower(), 3))


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Accumulate usage from one LLM call into a running total."""
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)
    if usage.get("cost"):
        total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])


class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: List[Dict[str, Any]], model: str, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pass


class GeminiClient(LLMProvider):
    """Wrapper for Google Gemini API with key rotation."""

    def __init__(self, keys_file: str = "state/gemini_keys.txt"):
        self.keys_file = keys_file
        self._keys = self._load_keys()
        self._current_idx = random.randint(0, len(self._keys) - 1) if self._keys else 0

    def _load_keys(self) -> List[str]:
        keys = []
        if os.path.exists(self.keys_file):
            with open(self.keys_file, "r") as f:
                keys = [k.strip() for k in f.read().replace(",", "\n").splitlines() if k.strip()]
        env_key = os.environ.get("GEMINI_API_KEY")
        if env_key and env_key not in keys:
            keys.append(env_key)
        return keys

    def _rotate_key(self):
        if not self._keys:
            return
        self._current_idx = (self._current_idx + 1) % len(self._keys)
        import google.generativeai as genai
        genai.configure(api_key=self._keys[self._current_idx])
        log.info(f"Rotated to Gemini key index {self._current_idx}")

    def chat(self, messages: List[Dict[str, Any]], model: str, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        import google.generativeai as genai
        
        retries = 3
        while retries > 0:
            try:
                # Re-configure with current key
                if self._keys:
                    genai.configure(api_key=self._keys[self._current_idx])
                
                # Strip model prefix
                actual_model = model.replace("gemini/", "", 1)
                g_model = genai.GenerativeModel(actual_model)
                
                gemini_messages = []
                for msg in messages:
                    role = "user" if msg["role"] == "user" else "model"
                    content = msg["content"]
                    if isinstance(content, str):
                        parts = [content]
                    else:
                        parts = [c["text"] for c in content if c.get("type") == "text"]
                    gemini_messages.append({"role": role, "parts": parts})
                    
                chat = g_model.start_chat(history=gemini_messages[:-1])
                resp = chat.send_message(gemini_messages[-1]["parts"])
                
                msg = {"role": "assistant", "content": resp.text}
                meta = resp.candidates[0].usage_metadata if resp.candidates else None
                usage = {
                    "prompt_tokens": meta.prompt_token_count if meta else 0,
                    "completion_tokens": meta.candidates_token_count if meta else 0,
                    "total_tokens": meta.total_token_count if meta else 0,
                    "cost": 0.0,
                }
                return msg, usage
                
            except Exception as e:
                # Catch 429 and rotate
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    log.warning(f"Gemini 429, rotating key. Retries left: {retries-1}")
                    self._rotate_key()
                    retries -= 1
                    time.sleep(1) # Backoff
                else:
                    raise e
        raise Exception("Gemini API exhausted after retries.")


class OpenRouterProvider(LLMProvider):
    """Wrapper for OpenRouter API."""

    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"

    def chat(self, messages: List[Dict[str, Any]], model: str, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        import requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://ouroboros.dev",
            "X-Title": "Ouroboros",
        }
        
        # Strip model prefix
        actual_model = model.replace("openrouter/", "", 1)
        
        payload = {
            "model": actual_model,
            "messages": messages,
        }
        
        resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        
        choice = data["choices"][0]
        msg = {"role": "assistant", "content": choice["message"]["content"]}
        usage = {
            "prompt_tokens": data["usage"]["prompt_tokens"],
            "completion_tokens": data["usage"]["completion_tokens"],
            "total_tokens": data["usage"]["total_tokens"],
            "cost": 0.0,
        }
        return msg, usage


class LLMClient:
    """Ouroboros Router client."""

    def __init__(self, *args, **kwargs):
        self._providers = {
            "gemini": GeminiClient(),
            "openrouter": OpenRouterProvider(),
        }

    def _get_provider(self, model: str) -> LLMProvider:
        prefix = model.split("/")[0]
        if prefix in self._providers:
            return self._providers[prefix]
        return self._providers["gemini"]

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384,
        tool_choice: str = "auto",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        provider = self._get_provider(model)
        return provider.chat(messages, model, tools=tools, reasoning_effort=reasoning_effort)

    def vision_query(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
        reasoning_effort: str = "low",
    ) -> Tuple[str, Dict[str, Any]]:
        """Send a vision query to an LLM."""
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            if "url" in img:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img["url"]},
                })
            elif "base64" in img:
                mime = img.get("mime", "image/png")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img['base64']}"},
                })
        
        messages = [{"role": "user", "content": content}]
        response_msg, usage = self.chat(
            messages=messages,
            model=model,
            tools=None,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        text = response_msg.get("content") or ""
        return text, usage

    def default_model(self) -> str:
        return os.environ.get("OUROBOROS_MODEL", DEFAULT_MODEL)

    def available_models(self) -> List[str]:
        return ["gemini/gemini-1.5-flash", "openrouter/gpt-oss-120b"]
