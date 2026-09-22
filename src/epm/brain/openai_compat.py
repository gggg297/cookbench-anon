from __future__ import annotations

"""
Helpers for OpenAI-compatible HTTP endpoints.

Different providers expect different base URL / path combinations:
- Some want "https://host" and the client appends "/v1/...".
- Some want "https://host/v1" and the client should NOT append another "/v1".
- Some expose OpenAI-like request bodies but use a non-/v1 path, such as
  "/api/paas/v4/chat/completions".

This module keeps URL joining and request-shape quirks consistent across EPM.
"""


DEFAULT_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
DEFAULT_MODELS_PATH = "/v1/models"


def join_openai_compatible_url(base_url: str, path: str) -> str:
    """
    Join an OpenAI-compatible base_url with a path like "/v1/chat/completions".

    If base_url already ends with "/v1" and the path starts with "/v1/",
    we avoid duplicating it.
    """

    b = (base_url or "").strip().rstrip("/")
    p = (path or "").strip()
    if not p.startswith("/"):
        p = "/" + p
    if b.endswith("/v1") and p.startswith("/v1/"):
        p = p[len("/v1") :]
    return b + p


def normalize_openai_compatible_path(path: str, *, default: str) -> str:
    p = (path or "").strip() or default
    if not p.startswith("/"):
        p = "/" + p
    return p


def uses_openai_reasoning_token_field(model: str) -> bool:
    """
    GPT-5 reasoning models reject `max_tokens` and require
    `max_completion_tokens` on chat/completions.
    """

    normalized = (model or "").strip().lower()
    return normalized.startswith("gpt-5")


def build_openai_token_limit_payload(*, model: str, max_tokens: int) -> dict[str, int]:
    value = int(max_tokens)
    if uses_openai_reasoning_token_field(model):
        return {"max_completion_tokens": value}
    return {"max_tokens": value}
