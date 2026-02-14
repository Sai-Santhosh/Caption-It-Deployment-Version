"""
Caption-It Gradio App
Deploy to Hugging Face Spaces - uses Inference API or local pipeline.
"""
import os

import gradio as gr

# Use HF Inference API by default (no GPU needed for free tier)
USE_INFERENCE_API = os.environ.get("USE_INFERENCE_API", "true").lower() == "true"
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "nlpconnect/vit-gpt2-image-captioning")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

if USE_INFERENCE_API:
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=HF_TOKEN if HF_TOKEN else None)

    def caption_image(image):
        if image is None:
            return ""
        result = client.image_to_text(image, model=HF_MODEL_ID)
        if isinstance(result, list) and result:
            r = result[0]
            return getattr(r, "generated_text", None) or (r.get("generated_text", "") if isinstance(r, dict) else str(r))
        return getattr(result, "generated_text", str(result)) if result else ""
else:
    from transformers import pipeline
    pipe = pipeline("image-to-text", model=HF_MODEL_ID, device=0)

    def caption_image(image):
        if image is None:
            return ""
        out = pipe(image)
        if isinstance(out, list) and out:
            return out[0].get("generated_text", "")
        return str(out)


demo = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Textbox(label="Caption", lines=3),
    title="Caption-It",
    description="Generate captions for images using ViT-GPT2. Upload an image or paste from clipboard.",
    examples=[
        ["https://images.pexels.com/photos/1054655/pexels-photo-1054655.jpeg"],
    ],
    article="""
## About
Caption-It uses a Vision Transformer (ViT) encoder with GPT-2 decoder for image captioning,
fine-tuned on Flickr8k. Part of the Caption-It project for Sieve-style AI data pipelines.

- **Model**: [nlpconnect/vit-gpt2-image-captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)
- **API**: Use `/api/v1/caption` for programmatic access
    """,
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
