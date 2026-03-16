"""
Ouroboros - LLM client.

Supports multiple LLM providers with dynamic routing.
Contract: chat(), default_model(), available_models(), add_usage().
"""

from __future__ import annotations

import logging
import os
import pathlib
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Primary engine
DEFAULT_MODEL = "gemini/gemini-1.5-flash"
DEFAULT_LIGHT_MODEL = DEFAULT_MODEL
GEMINI_MODEL_PREFIXES = ("gemini/", "google/")


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


def load_gemini_keys(keys_file: str = "state/gemini_keys.txt") -> List[str]:
    """Load Gemini API keys from env vars and/or a text file."""
    keys: List[str] = []

    def _append(raw: Optional[str]) -> None:
        if not raw:
            return
        for key in str(raw).replace(",", "\n").splitlines():
            cleaned = key.strip()
            if cleaned and cleaned not in keys:
                keys.append(cleaned)

    _append(os.environ.get("GEMINI_API_KEYS"))
    _append(os.environ.get("GEMINI_API_KEY"))

    path = pathlib.Path(keys_file)
    if path.exists():
        _append(path.read_text(encoding="utf-8"))

    return keys


class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: List[Dict[str, Any]], model: str, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pass


class GeminiClient(LLMProvider):
    """Wrapper for Google Gemini API with key rotation."""

    def __init__(self, keys_file: str = "state/gemini_keys.txt"):
        self.keys_file = keys_file
        self._keys = load_gemini_keys(keys_file)
        self._current_idx = random.randint(0, len(self._keys) - 1) if self._keys else 0

    def _require_keys(self) -> None:
        if not self._keys:
            raise RuntimeError(
                "Gemini is selected but no keys are configured. "
                "Set GEMINI_API_KEY or GEMINI_API_KEYS, or create state/gemini_keys.txt."
            )

    @staticmethod
    def _normalize_model_name(model: str) -> str:
        model = str(model or "").strip()
        for prefix in GEMINI_MODEL_PREFIXES:
            if model.startswith(prefix):
                return model[len(prefix):]
        return model or DEFAULT_MODEL.split("/", 1)[1]

    @staticmethod
    def _extract_text_parts(content: Any) -> List[str]:
        if isinstance(content, str):
            return [content]
        if not isinstance(content, list):
            return [str(content)]
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return parts

    def _rotate_key(self) -> None:
        self._require_keys()
        self._current_idx = (self._current_idx + 1) % len(self._keys)
        import google.generativeai as genai

        genai.configure(api_key=self._keys[self._current_idx])
        log.info("Rotated to Gemini key index %s", self._current_idx)

    def chat(self, messages: List[Dict[str, Any]], model: str, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._require_keys()
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError(
                "Gemini support requires google-generativeai. Install dependencies from requirements.txt."
            ) from exc

        retries = 3
        while retries > 0:
            try:
                genai.configure(api_key=self._keys[self._current_idx])
                actual_model = self._normalize_model_name(model)
                g_model = genai.GenerativeModel(actual_model)

                gemini_messages = []
                for msg in messages:
                    role = "user" if msg["role"] == "user" else "model"
                    parts = self._extract_text_parts(msg.get("content"))
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
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    log.warning("Gemini 429, rotating key. Retries left: %s", retries - 1)
                    self._rotate_key()
                    retries -= 1
                    time.sleep(1)
                else:
                    raise
        raise RuntimeError("Gemini API exhausted after retries.")


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
            "google": GeminiClient(),
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
        return [
            "gemini/gemini-1.5-flash",
            "google/gemini-1.5-flash",
            "openrouter/gpt-oss-120b",
        ]
