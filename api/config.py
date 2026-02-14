"""
Production configuration for Caption-It API.
Loads from environment variables with sensible defaults.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # Inference mode: "local" loads PyTorch model; "huggingface" uses HF Inference API
    inference_mode: Literal["local", "huggingface"] = Field(
        default="huggingface",
        description="Inference backend: local (PyTorch) or huggingface (HF API)",
    )

    # Local model (when inference_mode=local)
    model_path: str = Field(
        default="./models/caption-it",
        description="Path to local model directory (config.json, pytorch_model.bin)",
    )

    # Hugging Face (when inference_mode=huggingface)
    hf_model_id: str = Field(
        default="nlpconnect/vit-gpt2-image-captioning",
        description="Hugging Face model ID for Inference API",
    )
    hf_api_token: str | None = Field(
        default=None,
        description="Hugging Face API token (optional for public models)",
    )
    hf_api_url: str | None = Field(
        default=None,
        description="Override HF Inference API base URL (e.g. for Enterprise)",
    )

    # Generation parameters
    max_seq_len: int = Field(default=24, ge=8, le=128, description="Max caption length")
    num_beams: int = Field(default=4, ge=1, le=10, description="Beam search width")
    length_penalty: float = Field(default=1.2, ge=0.0, le=2.0)
    no_repeat_ngram_size: int = Field(default=4, ge=0, le=10)

    # API settings
    api_title: str = Field(default="Caption-It API", description="OpenAPI title")
    api_version: str = Field(default="1.0.0", description="API version")
    cors_origins: str = Field(
        default="*",
        description="Comma-separated CORS origins",
    )
    max_upload_mb: int = Field(default=10, ge=1, le=50, description="Max upload size MB")
    request_timeout_seconds: int = Field(default=30, ge=5, le=120)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
