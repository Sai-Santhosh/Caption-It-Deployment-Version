"""
Local PyTorch model inference for Caption-It.
Supports both standard VisionEncoderDecoderModel and custom ViT-GPT2
with MLP bridge + extra decoder blocks (Flickr8k fine-tuned).
"""
from __future__ import annotations

import io
import time
from pathlib import Path
from typing import BinaryIO

import torch
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor

from .base import BaseCaptioner, CaptionResult


# Default architecture constants (matches your training)
ENCODER_NAME = "google/vit-base-patch16-224-in21k"
DECODER_NAME = "gpt2"
NEW_GPT2_BLOCKS = 2


def _patch_encoder_projection(model: VisionEncoderDecoderModel) -> None:
    """Wrap encoder so its output is projected through enc_to_dec_proj before decoder."""
    if not hasattr(model, "enc_to_dec_proj"):
        return

    from transformers.modeling_outputs import BaseModelOutput

    original_encoder = model.encoder
    proj = model.enc_to_dec_proj

    class ProjEncoder(torch.nn.Module):
        def __init__(self, enc, projection):
            super().__init__()
            self.encoder = enc
            self.projection = projection

        def forward(self, *args, **kwargs):
            out = self.encoder(*args, **kwargs)
            projected = self.projection(out.last_hidden_state)
            return BaseModelOutput(
                last_hidden_state=projected,
                hidden_states=out.hidden_states,
                attentions=out.attentions,
            )

    model.encoder = ProjEncoder(original_encoder, proj)
    # Now encoder and proj are combined; decoder gets projected output
    # The default forward will use encoder_outputs[0] which is now projected


def _load_image(image_source: bytes | BinaryIO) -> Image.Image:
    """Load PIL Image from bytes or file-like object."""
    if isinstance(image_source, bytes):
        return Image.open(io.BytesIO(image_source)).convert("RGB")
    return Image.open(image_source).convert("RGB")


class LocalCaptioner(BaseCaptioner):
    """
    Local ViT-GPT2 captioner.
    Loads from Hugging Face format (config.json, pytorch_model.bin) or
    builds standard nlpconnect architecture if custom weights not found.
    """

    def __init__(
        self,
        model_path: str,
        max_seq_len: int = 24,
        num_beams: int = 4,
        length_penalty: float = 1.2,
        no_repeat_ngram_size: int = 4,
        device: str | None = None,
    ):
        self.model_path = Path(model_path)
        self.max_seq_len = max_seq_len
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.processor = None
        self.tokenizer = None
        self.model_id = str(self.model_path)
        self._load_model()

    def _load_model(self) -> None:
        """Load model, processor, and tokenizer."""
        has_weights = (self.model_path / "pytorch_model.bin").exists() or (
            self.model_path / "model.safetensors"
        ).exists()

        if has_weights:
            # Load custom fine-tuned model
            self._load_custom_model()
        else:
            # Fallback: standard nlpconnect model from HF
            self._load_standard_model()

        self.model.eval()
        self.model.to(self.device)

    def _load_standard_model(self) -> None:
        """Load standard nlpconnect/vit-gpt2-image-captioning from HF."""
        model_id = "nlpconnect/vit-gpt2-image-captioning"
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self.processor = ViTImageProcessor.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model_id = model_id
        self._configure_generation()

    def _load_custom_model(self) -> None:
        """Load custom ViT-GPT2 with MLP bridge and extra blocks."""
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block

        # Build base architecture
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            ENCODER_NAME,
            DECODER_NAME,
            tie_encoder_decoder=False,
        )

        dec_conf = model.decoder.config
        enc_size = model.encoder.config.hidden_size
        dec_size = dec_conf.n_embd

        # Custom MLP bridge (matches training) - replaces default linear when dims match
        model.enc_to_dec_proj = torch.nn.Sequential(
            torch.nn.Linear(enc_size, dec_size),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(dec_size, dec_size),
        )

        # Extra GPT-2 blocks
        for _ in range(NEW_GPT2_BLOCKS):
            model.decoder.transformer.h.append(GPT2Block(dec_conf))

        new_n = len(model.decoder.transformer.h)
        dec_conf.n_layer = new_n
        model.decoder.transformer.n_layer = new_n
        model.decoder.transformer.config.n_layer = new_n
        model.decoder.config.n_layer = new_n

        # Load weights (only matching keys - handles partial checkpoints)
        weight_path = self.model_path / "pytorch_model.bin"
        if not weight_path.exists():
            weight_path = self.model_path / "model.safetensors"

        if weight_path.exists():
            if str(weight_path).endswith(".bin"):
                state = torch.load(weight_path, map_location="cpu", weights_only=True)
            else:
                from safetensors.torch import load_file

                state = load_file(str(weight_path))
            model.load_state_dict(state, strict=False)

        # Patch forward to always use enc_to_dec_proj (custom MLP vs default linear)
        self._patch_encoder_projection(model)

        self.model = model
        self.processor = ViTImageProcessor.from_pretrained(ENCODER_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(DECODER_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_id = str(self.model_path)
        self._configure_generation()

    def _configure_generation(self) -> None:
        """Set generation config on model."""
        cfg = self.model.config
        cfg.decoder_start_token_id = self.tokenizer.bos_token_id
        cfg.eos_token_id = self.tokenizer.eos_token_id
        cfg.pad_token_id = self.tokenizer.pad_token_id
        cfg.max_length = self.max_seq_len

    def caption_image(self, image_source: bytes | BinaryIO) -> CaptionResult:
        img = _load_image(image_source)
        pixel_values = self.processor(img, return_tensors="pt").pixel_values.to(
            self.device
        )

        start = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                pixel_values,
                max_length=self.max_seq_len,
                num_beams=self.num_beams,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                length_penalty=self.length_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        caption = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        return CaptionResult(
            caption=caption,
            model_id=self.model_id,
            inference_time_ms=elapsed_ms,
        )

    def is_ready(self) -> bool:
        return self.model is not None
