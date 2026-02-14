"""
Caption-It inference module.
Provides both local PyTorch and Hugging Face API backends.
"""
from .base import BaseCaptioner, CaptionResult
from .factory import get_captioner

__all__ = ["BaseCaptioner", "CaptionResult", "get_captioner"]
