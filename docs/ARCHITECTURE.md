# Caption-It: System Architecture & Model Design

> **Technical Architecture Document** — Vision-Language Transformer for Image Captioning

---

## 1. High-Level Architecture Overview

Caption-It implements a **dual-transformer encoder–decoder** architecture with cross-modal attention, combining a Vision Transformer (ViT) encoder with a GPT-2 language model decoder. The system bridges visual and linguistic representations through a learned MLP projection and cross-attention layers in every decoder block.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Caption-It Production Architecture                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────────────────────────────┐   │
│   │  Image   │───▶│  ViT-Base    │───▶│  2-Layer MLP Bridge (768→768)        │   │
│   │ 224×224  │    │  12 Blocks   │    │  Linear → Tanh → Dropout → Linear     │   │
│   └──────────┘    └──────────────┘    └────────────────┬─────────────────────┘   │
│                                                         │                        │
│                                                         ▼                        │
│   ┌──────────────────────────────────────────────────────────────────────────┐   │
│   │  GPT-2 Decoder (12 original + 2 new blocks)                               │   │
│   │  Each block: Self-Attention ← Cross-Attention(encoder_output) → FFN       │   │
│   └──────────────────────────────────────────────────────────────────────────┘   │
│                                         │                                         │
│                                         ▼                                         │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                         │
│   │ Beam Search  │   │ No-Repeat    │   │ Length       │                         │
│   │ (k=1,3,5,10) │   │ N-gram (5)  │   │ Penalty 1.6  │                         │
│   └──────────────┘   └──────────────┘   └──────────────┘                         │
│                                         │                                         │
│                                         ▼                                         │
│                              Generated Caption (text)                             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1 Vision Encoder (ViT-Base)

| Parameter       | Value                    |
|-----------------|--------------------------|
| Model           | `google/vit-base-patch16-224-in21k` |
| Input Resolution| 224×224 RGB               |
| Patch Size      | 16×16 → 196 patches       |
| Hidden Size     | 768                       |
| Layers          | 12 Transformer blocks    |
| Attention Heads | 12                        |
| Pre-training    | ImageNet-21K              |

The encoder produces a sequence of 197 vectors (196 patch tokens + 1 [CLS] token), each of dimension 768.

### 2.2 MLP Bridge

The bridge projects encoder output to decoder dimensionality and adds non-linear capacity:

```python
enc_to_dec_proj = nn.Sequential(
    nn.Linear(768, 768),   # Encoder → Decoder dimension alignment
    nn.Tanh(),
    nn.Dropout(0.1),
    nn.Linear(768, 768),
)
```

### 2.3 Language Decoder (GPT-2 Extended)

| Parameter       | Value                    |
|-----------------|--------------------------|
| Base Model      | GPT-2 Small              |
| Original Layers | 12                       |
| Additional Layers| 2 (newly initialized)     |
| Total Blocks    | 14                       |
| Hidden Size     | 768                      |
| Vocab Size      | 50,257 (GPT-2 BPE)       |
| Max Length      | 24 tokens (configurable) |

Each decoder block contains:
- **Self-attention** over previously generated tokens
- **Cross-attention** over encoder/bridge output (Q from decoder, K/V from encoder)
- **Feed-forward** network

### 2.4 Training Strategy: Progressive Unfreezing

| Phase   | Epochs   | Encoder State       | Decoder State |
|---------|----------|---------------------|---------------|
| Phase 1 | 0–2      | Frozen              | Trainable     |
| Phase 2 | 2–8      | Last N blocks unfrozen | Trainable  |

Encoder LR: 5e-6 (decayed to 1.5e-6 after unfreeze)  
Decoder LR: 2e-5

---

## 3. Inference Pipeline Architecture

### 3.1 Production Stack

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────────────────────┐
│   Frontend  │────▶│   NGINX     │────▶│  FastAPI API (port 8000)        │
│   (Vite)    │     │   (proxy)   │     │  - /api/v1/caption (upload)     │
│   port 80   │     └─────────────┘     │  - /api/v1/caption/url           │
└─────────────┘                        │  - /api/v1/health               │
                                        └──────────────┬──────────────────┘
                                                       │
                          ┌────────────────────────────┼────────────────────────────┐
                          │                            │                            │
                          ▼                            ▼                            ▼
                ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
                │ HuggingFace API  │       │ Local Model      │       │ Gradio App        │
                │ (Serverless)     │       │ (PyTorch)        │       │ (HF Spaces)       │
                └──────────────────┘       └──────────────────┘       └──────────────────┘
```

### 3.2 Inference Modes

| Mode           | Use Case                | Requirements                    |
|----------------|--------------------------|---------------------------------|
| `huggingface`  | Zero-GPU deployment      | HF API token (optional)         |
| `local`        | On-premise, offline      | PyTorch, GPU recommended        |

---

## 4. Data Flow

1. **Upload/URL** → Image bytes received by API
2. **Preprocessing** → Resize to 224×224, normalize (ImageNet stats)
3. **Encoder** → ViT produces 197×768 hidden states
4. **Bridge** → MLP projects to decoder space
5. **Decoder** → Autoregressive generation with beam search
6. **Decoding** → BPE decode → Final caption string

---

## 5. Deployment Architecture

Docker Compose orchestrates:

- **api**: Python 3.11-slim, FastAPI, Uvicorn
- **frontend**: Nginx serving static Vite build, proxies `/api` to backend

Volumes: `./models` mounted for local model weights.

---

*For the visual cross-modal flow diagram, see `docs/assets/crossmodal.pdf` or `docs/assets/architecture.pdf`.*
