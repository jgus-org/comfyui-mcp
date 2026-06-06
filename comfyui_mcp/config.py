"""Env-driven runtime configuration for the ComfyUI MCP server."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Config:
    comfyui_url: str
    host: str
    port: int
    # Writable bind-mount of ComfyUI's models dir, shared with ComfyUI; its filename cache tracks the dir mtime, so files dropped here are picked up automatically.
    models_dir: str
    # OpenAI-compatible vision endpoint (model ids are per-profile in server.py).
    vision_base_url: str
    vision_api_key: str
    # HuggingFace token; sent as a bearer to fetch gated base models (Flux.1-dev).
    hf_token: str

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            comfyui_url=os.environ["COMFYUI_URL"],
            host=os.environ.get("HOST", "::"),
            port=int(os.environ.get("PORT", "8189")),
            models_dir=os.environ["COMFYUI_MODELS_DIR"],
            vision_base_url=os.environ["VISION_API_BASE_URL"],
            vision_api_key=os.environ["VISION_API_KEY"],
            hf_token=os.environ["HF_TOKEN"],
        )
