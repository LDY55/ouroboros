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
import uuid
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
        log.info("Rotated to Gemini key index %s", self._current_idx)

    @staticmethod
    def _build_google_genai_contents(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        contents: List[Dict[str, Any]] = []
        tool_name_by_id: Dict[str, str] = {}
        for msg in messages:
            role = str(msg.get("role") or "")
            if role == "system":
                continue

            if role == "assistant" and msg.get("_gemini_content") is not None:
                contents.append(msg["_gemini_content"])
                for tool_call in msg.get("tool_calls") or []:
                    fn = ((tool_call or {}).get("function") or {})
                    fn_name = str(fn.get("name") or "")
                    call_id = str((tool_call or {}).get("id") or "")
                    if fn_name and call_id:
                        tool_name_by_id[call_id] = fn_name
                continue

            if role == "tool":
                tool_call_id = str(msg.get("tool_call_id") or "")
                tool_name = tool_name_by_id.get(tool_call_id, "tool_result")
                content = str(msg.get("content") or "")
                parts = [{
                    "function_response": {
                        "name": tool_name,
                        "response": {"result": content},
                    }
                }]
                contents.append({"role": "user", "parts": parts})
                continue

            parts = [{"text": text} for text in GeminiClient._extract_text_parts(msg.get("content")) if text]
            for tool_call in msg.get("tool_calls") or []:
                fn = ((tool_call or {}).get("function") or {})
                fn_name = str(fn.get("name") or "")
                call_id = str((tool_call or {}).get("id") or uuid.uuid4().hex)
                tool_name_by_id[call_id] = fn_name
                try:
                    fn_args = fn.get("arguments") or "{}"
                except Exception:
                    fn_args = "{}"
                parts.append({
                    "function_call": {
                        "name": fn_name,
                        "args": fn_args if isinstance(fn_args, dict) else _safe_json_loads(fn_args),
                    }
                })
            if parts:
                mapped_role = "user" if role == "user" else "model"
                contents.append({"role": mapped_role, "parts": parts})
        return contents

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if text:
            return str(text)

        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            texts = []
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    texts.append(str(part_text))
            if texts:
                return "\n".join(texts).strip()
        return ""

    @staticmethod
    def _extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
        tool_calls: List[Dict[str, Any]] = []
        direct_calls = getattr(response, "function_calls", None) or []
        for function_call in direct_calls:
            call_name = str(getattr(function_call, "name", "") or "")
            call_args = getattr(function_call, "args", {}) or {}
            call_id = str(
                getattr(function_call, "id", "")
                or getattr(function_call, "call_id", "")
                or f"call_{uuid.uuid4().hex[:12]}"
            )
            tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": call_name,
                    "arguments": _safe_json_dumps(call_args),
                },
            })
        if tool_calls:
            return tool_calls

        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                function_call = getattr(part, "function_call", None)
                if not function_call:
                    continue
                call_name = str(getattr(function_call, "name", "") or "")
                call_args = getattr(function_call, "args", {}) or {}
                call_id = str(
                    getattr(function_call, "id", "")
                    or getattr(function_call, "call_id", "")
                    or f"call_{uuid.uuid4().hex[:12]}"
                )
                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": call_name,
                        "arguments": _safe_json_dumps(call_args),
                    },
                })
        return tool_calls

    @staticmethod
    def _extract_system_instruction(messages: List[Dict[str, Any]]) -> str:
        chunks = []
        for msg in messages:
            if str(msg.get("role") or "") != "system":
                continue
            text_parts = GeminiClient._extract_text_parts(msg.get("content"))
            if text_parts:
                chunks.append("\n".join(text_parts))
        return "\n\n".join(chunk for chunk in chunks if chunk.strip())

    @staticmethod
    def _convert_tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        converted = []
        for tool in tools or []:
            fn = (tool or {}).get("function") or {}
            fn_name = str(fn.get("name") or "")
            if not fn_name:
                continue
            converted.append({
                "function_declarations": [{
                    "name": fn_name,
                    "description": str(fn.get("description") or ""),
                    "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                }]
            })
        return converted

    @staticmethod
    def _extract_usage(response: Any) -> Dict[str, Any]:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
            }
        return {
            "prompt_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
            "completion_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
            "total_tokens": int(getattr(usage, "total_token_count", 0) or 0),
            "cost": 0.0,
        }

    @staticmethod
    def _extract_debug_meta(response: Any) -> Dict[str, Any]:
        candidates = getattr(response, "candidates", None) or []
        function_calls = getattr(response, "function_calls", None) or []
        finish_reasons = []
        for candidate in candidates:
            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason is not None:
                finish_reasons.append(str(finish_reason))
        return {
            "candidate_count": len(candidates),
            "function_call_count": len(function_calls),
            "finish_reasons": finish_reasons,
            "response_text_present": bool(getattr(response, "text", None)),
        }

    def chat(self, messages: List[Dict[str, Any]], model: str, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._require_keys()
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Gemini support requires google-genai. Install dependencies from requirements.txt."
            ) from exc

        retries = 3
        while retries > 0:
            try:
                actual_model = self._normalize_model_name(model)
                client = genai.Client(api_key=self._keys[self._current_idx])
                contents = self._build_google_genai_contents(messages)
                if not contents:
                    raise RuntimeError("Gemini request contained no text parts.")
                config = types.GenerateContentConfig(
                    system_instruction=self._extract_system_instruction(messages) or None,
                    tools=self._convert_tools(kwargs.get("tools")),
                )

                resp = client.models.generate_content(
                    model=actual_model,
                    contents=contents,
                    config=config,
                )

                text = self._extract_response_text(resp)
                tool_calls = self._extract_tool_calls(resp)
                msg: Dict[str, Any] = {"role": "assistant", "content": text}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                candidates = getattr(resp, "candidates", None) or []
                if candidates:
                    content = getattr(candidates[0], "content", None)
                    if content is not None:
                        msg["_gemini_content"] = content
                msg["_gemini_debug"] = self._extract_debug_meta(resp)
                usage = self._extract_usage(resp)
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


def _safe_json_loads(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        import json
        loaded = json.loads(str(value or "{}"))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _safe_json_dumps(value: Any) -> str:
    try:
        import json
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return "{}"
