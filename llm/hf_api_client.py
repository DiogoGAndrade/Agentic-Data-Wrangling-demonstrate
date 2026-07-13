from __future__ import annotations

import os
from typing import Any

import streamlit as st
from huggingface_hub import InferenceClient


class HuggingFaceAPIClient:
    """Client compatible with the application's existing generate() call."""

    def __init__(self, token: str, model: str = "Qwen/Qwen2.5-3B-Instruct", provider: str = "auto", max_tokens: int = 1200) -> None:
        if not token:
            raise ValueError("HF_TOKEN is missing. Add it in the Streamlit Cloud app secrets.")
        self.model = model
        self.provider = provider
        self.max_tokens = max_tokens
        self.client = InferenceClient(provider=provider, api_key=token)

    @classmethod
    def from_streamlit_secrets(cls) -> "HuggingFaceAPIClient":
        def _read(name: str, default: str | None = None) -> str | None:
            try:
                value: Any = st.secrets.get(name, default)
            except Exception:
                value = os.getenv(name, default)
            return str(value) if value is not None else None
        return cls(
            token=_read("HF_TOKEN", "") or "",
            model=_read("HF_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct") or "Qwen/Qwen2.5-3B-Instruct",
            provider=_read("HF_PROVIDER", "auto") or "auto",
            max_tokens=int(_read("HF_MAX_TOKENS", "1200") or "1200"),
        )

    def generate(self, prompt: str, system: str = "", temperature: float = 0.2, max_tokens: int | None = None) -> str:
        try:
            response = self.client.chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system or "You are a data-wrangling assistant. Return only valid JSON matching the requested schema."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(
                "The external model request failed. This may be caused by provider availability, exhausted credits, model availability, or an invalid token. "
                f"Original error: {exc}"
            ) from exc
        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError("The external model returned no choices.")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message else None
        if not content:
            raise RuntimeError("The external model returned an empty response.")
        return str(content).strip()
