"""
Hugging Face Inference API client for image captioning.
Uses the HF Inference API - no local GPU required.
"""
import base64
import io
import time
from typing import BinaryIO

import httpx

from .base import BaseCaptioner, CaptionResult


# HF Inference API base URL
HF_INFERENCE_URL = "https://api-inference.huggingface.co/models"


class HuggingFaceCaptioner(BaseCaptioner):
    """Caption images via Hugging Face Inference API."""

    def __init__(
        self,
        model_id: str,
        api_token: str | None = None,
        api_url: str | None = None,
        timeout: float = 30.0,
    ):
        self.model_id = model_id
        self.api_token = api_token
        self.base_url = api_url or f"{HF_INFERENCE_URL}/{model_id}"
        self.timeout = timeout
        self._headers: dict[str, str] = {}
        if api_token:
            self._headers["Authorization"] = f"Bearer {api_token}"

    def _get_image_bytes(self, source: bytes | BinaryIO) -> bytes:
        if isinstance(source, bytes):
            return source
        return source.read()

    def caption_image(self, image_source: bytes | BinaryIO) -> CaptionResult:
        img_bytes = self._get_image_bytes(image_source)
        if not img_bytes:
            raise ValueError("Empty image data")

        start = time.perf_counter()
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                self.base_url,
                headers=self._headers,
                content=img_bytes,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        resp.raise_for_status()
        data = resp.json()

        # HF image-to-text returns list of dicts: [{"generated_text": "..."}]
        if isinstance(data, list):
            caption = data[0].get("generated_text", "") if data else ""
        elif isinstance(data, dict):
            caption = data.get("generated_text", data.get("caption", ""))
        else:
            caption = str(data)

        return CaptionResult(
            caption=caption.strip(),
            model_id=self.model_id,
            inference_time_ms=elapsed_ms,
        )

    def is_ready(self) -> bool:
        """HF API is always ready (no model loading)."""
        return True
