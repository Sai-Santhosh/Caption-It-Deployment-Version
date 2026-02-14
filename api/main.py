"""
Caption-It Production API
FastAPI backend for image captioning via local model or Hugging Face Inference API.
"""
from __future__ import annotations

import logging
from io import BytesIO

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

from config import get_settings
from inference import get_captioner
from inference.base import CaptionResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded captioner
_captioner = None


def get_captioner_instance():
    """Lazy-load captioner on first request."""
    global _captioner
    if _captioner is None:
        _captioner = get_captioner(
            inference_mode=settings.inference_mode,
            model_path=settings.model_path,
            hf_model_id=settings.hf_model_id,
            hf_api_token=settings.hf_api_token,
            hf_api_url=settings.hf_api_url,
            max_seq_len=settings.max_seq_len,
            num_beams=settings.num_beams,
            request_timeout=settings.request_timeout_seconds,
        )
        logger.info("Captioner initialized: mode=%s", settings.inference_mode)
    return _captioner


# --- Schemas ---


class CaptionResponse(BaseModel):
    """API response for caption endpoint."""

    caption: str
    model_id: str
    inference_time_ms: float | None = None


class CaptionFromUrlRequest(BaseModel):
    """Request body for URL-based captioning."""

    url: HttpUrl


# --- Endpoints ---


@app.get("/api/v1/health")
async def health() -> dict:
    """Health check."""
    captioner = get_captioner_instance()
    return {
        "status": "ok",
        "inference_mode": settings.inference_mode,
        "model_ready": captioner.is_ready(),
    }


@app.post("/api/v1/caption", response_model=CaptionResponse)
async def caption_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG)"),
) -> CaptionResponse:
    """
    Generate caption from uploaded image.
    Supports JPEG, PNG. Max size: 10MB.
    """
    content_type = file.content_type or ""
    if "image" not in content_type and not (file.filename and "." in file.filename):
        raise HTTPException(
            status_code=400,
            detail="Expected image file (JPEG, PNG)",
        )

    try:
        data = await file.read()
    except Exception as e:
        logger.exception("Failed to read upload")
        raise HTTPException(status_code=400, detail="Failed to read image") from e

    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_upload_mb}MB",
        )

    captioner = get_captioner_instance()
    try:
        result = captioner.caption_image(BytesIO(data))
    except Exception as e:
        logger.exception("Caption generation failed")
        raise HTTPException(status_code=500, detail="Caption generation failed") from e

    return CaptionResponse(
        caption=result.caption,
        model_id=result.model_id,
        inference_time_ms=result.inference_time_ms,
    )


@app.post("/api/v1/caption/url", response_model=CaptionResponse)
async def caption_from_url(body: CaptionFromUrlRequest) -> CaptionResponse:
    """
    Generate caption from image URL.
    Fetches the image and runs inference.
    """
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            resp = await client.get(str(body.url))
        resp.raise_for_status()
        data = resp.content
    except httpx.HTTPError as e:
        logger.warning("Failed to fetch image URL: %s", e)
        raise HTTPException(status_code=400, detail="Failed to fetch image URL") from e

    if not data:
        raise HTTPException(status_code=400, detail="Empty image at URL")

    captioner = get_captioner_instance()
    try:
        result = captioner.caption_image(data)
    except Exception as e:
        logger.exception("Caption generation failed")
        raise HTTPException(status_code=500, detail="Caption generation failed") from e

    return CaptionResponse(
        caption=result.caption,
        model_id=result.model_id,
        inference_time_ms=result.inference_time_ms,
    )


@app.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {"service": "Caption-It API", "docs": "/api/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
