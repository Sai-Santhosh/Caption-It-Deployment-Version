"""
Base interface for caption inference.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import BinaryIO


@dataclass
class CaptionResult:
    """Result of image captioning."""

    caption: str
    model_id: str
    inference_time_ms: float | None = None


class BaseCaptioner(ABC):
    """Abstract base for caption inference backends."""

    @abstractmethod
    def caption_image(self, image_source: bytes | BinaryIO) -> CaptionResult:
        """Generate caption from image bytes. Raises on invalid image."""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """Whether the captioner is ready to serve requests."""
        ...
