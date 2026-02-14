"""Factory for creating the appropriate captioner based on config."""
from __future__ import annotations

from .base import BaseCaptioner
from .hf_client import HuggingFaceCaptioner
from .local_model import LocalCaptioner


def get_captioner(
    inference_mode: str = "huggingface",
    model_path: str = "./models/caption-it",
    hf_model_id: str = "nlpconnect/vit-gpt2-image-captioning",
    hf_api_token: str | None = None,
    hf_api_url: str | None = None,
    max_seq_len: int = 24,
    num_beams: int = 4,
    request_timeout: int = 30,
) -> BaseCaptioner:
    """
    Create captioner instance based on inference mode.

    Args:
        inference_mode: "local" or "huggingface"
        model_path: Path to local model (when mode=local)
        hf_model_id: Hugging Face model ID (when mode=huggingface)
        hf_api_token: HF API token for private models
        hf_api_url: Override HF Inference API URL
        max_seq_len: Max caption length
        num_beams: Beam search width
        request_timeout: Timeout in seconds (HF API)

    Returns:
        Configured BaseCaptioner instance
    """
    if inference_mode == "local":
        return LocalCaptioner(
            model_path=model_path,
            max_seq_len=max_seq_len,
            num_beams=num_beams,
        )
    return HuggingFaceCaptioner(
        model_id=hf_model_id,
        api_token=hf_api_token,
        api_url=hf_api_url,
        timeout=float(request_timeout),
    )
