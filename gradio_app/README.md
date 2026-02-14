# Caption-It Gradio App

Deploy to [Hugging Face Spaces](https://huggingface.co/spaces) in one click.

## Quick Deploy

1. Create a new Space at https://huggingface.co/new-space
2. Select **Gradio** as SDK
3. Clone this repo or copy `app.py` and `requirements.txt`
4. Push to your Space

## Configuration (Space Secrets)

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | Hugging Face token for gated models or higher rate limits |
| `USE_INFERENCE_API` | Set to `true` (default) for serverless - no GPU needed |
| `HF_MODEL_ID` | Model ID, default: `nlpconnect/vit-gpt2-image-captioning` |

With `USE_INFERENCE_API=true`, the app uses the free Inference API - no GPU required.

For custom fine-tuned models, set `HF_MODEL_ID` to your model ID and ensure it supports `image-to-text` pipeline.
