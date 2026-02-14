# Caption-It

**Production-grade image captioning** with Vision Transformer (ViT) + GPT-2 cross-modal attention. Fine-tuned on Flickr8k with progressive unfreezing and exhaustive ablations. Deploy via Docker, Hugging Face, or local PyTorch.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Highlights](#-highlights)
- [Architecture](#-architecture)
- [How It Works](#-how-it-works)
- [Tools & Tech Stack](#-tools--tech-stack)
- [Project Report & Findings](#-project-report--findings)
- [Installation & Quick Start](#-installation--quick-start)
- [Deployment](#-deployment)
- [API Reference](#-api-reference)
- [Research Details](#-research-details)
- [File Structure](#-file-structure)
- [References](#-references)

---

## Highlights

| Feature | Description |
|---------|-------------|
| **Transformer-based** | Full cross-modal attention between ViT encoder and GPT-2 decoder |
| **Progressive training** | Frozen → partially unfrozen → fully trainable ViT |
| **Metrics** | BLEU-4, ROUGE-L, CIDEr, METEOR, BERTScore |
| **Ablations** | Beam-size exploration, cross-attention ablation, attention diagnostics |
| **Decoding** | Greedy and beam search (k=1,3,5,10) with length penalty |

---

## Architecture

### System Overview

```
Image (224×224) → ViT-Base (12 blocks) → MLP Bridge (768→768) → GPT-2 (14 blocks) → Caption
                                        ↑
                              Cross-Attention in every decoder block
```

### Baselines

| Model | Description |
|-------|-------------|
| ResNet50 + LSTM | Frozen ResNet encoder → linear proj → 512-d LSTM |
| MobileNetV3 + GRU | Frozen MobileNetV3 → linear proj → 512-d GRU |

### Final Cross-Modal ViT-GPT2

| Component | Details |
|-----------|---------|
| **Encoder** | ViT-Base (12 blocks, 768-d), frozen initially |
| **Bridge** | 2-layer MLP (768→768): Linear → Tanh → Dropout → Linear |
| **Decoder** | GPT-2 small (12 original + 2 new transformer layers) |
| **Unfreezing** | ViT progressively unfrozen after epoch 2 |
| **Cross-Attention** | Enabled in every decoder layer (Q from decoder, K/V from encoder) |

### Architecture Diagrams

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full system architecture (export to PDF) |
| [docs/assets/crossmodal.pdf](docs/assets/crossmodal.pdf) | Cross-modal data flow |
| [docs/assets/archi1.pdf](docs/assets/archi1.pdf) | ViT-GPT2 architecture (project report) |

> 📄 **Export to PDF**: Use `pandoc docs/ARCHITECTURE.md -o docs/assets/architecture.pdf` or your preferred Markdown→PDF tool.

---

## How It Works

### 1. Training Pipeline

1. **Data**: Flickr8k (8,091 images × 5 captions) → Karpathy split (6k train / 1k val / 1k test)
2. **Preprocessing**: Resize 224×224, ImageNet normalization, NLTK tokenization, max length 30, vocab freq ≥5 → 2,984 tokens
3. **Model**: ViT encoder + MLP bridge + GPT-2 decoder with 2 extra blocks
4. **Strategy**: Label smoothing (ε=0.1), AdamW, cosine decay, 5% warm-up; ViT frozen for 2 epochs then unfrozen
5. **Decoding**: Beam search (k=4), length penalty 1.6, no-repeat 5-gram

### 2. Inference Flow

```
Upload/URL → FastAPI → Image bytes → ViT + Bridge → GPT-2 beam decode → Caption JSON
```

### 3. Modes

| Mode | Use Case | Setup |
|------|----------|-------|
| `huggingface` | Serverless, no GPU | `INFERENCE_MODE=huggingface` (default) |
| `local` | On-premise, offline | `INFERENCE_MODE=local`, place model in `./models/caption-it` |

---

## Tools & Tech Stack

| Layer | Technology |
|-------|------------|
| **ML Framework** | PyTorch 2.0+, Transformers 4.36+ |
| **Models** | ViT-Base (`google/vit-base-patch16-224-in21k`), GPT-2, HuggingFace Inference API |
| **API** | FastAPI, Uvicorn |
| **Frontend** | Vite, TypeScript, vanilla JS |
| **Gradio** | Hugging Face Spaces deployment |
| **Containers** | Docker, Docker Compose |
| **Metrics** | BLEU (sacrebleu), ROUGE (rouge_score), CIDEr (pycocoevalcap), BERTScore |

### Full Dependency List

**API**: `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`, `httpx`, `torch`, `transformers`, `accelerate`, `Pillow`, `safetensors`

**Research**: `evaluate`, `sacrebleu`, `rouge_score`, `bert_score`, `pycocoevalcap`, `datasets`, `tqdm`

---

## Project Report & Findings

### Report Documents

| Document | Path | Description |
|----------|------|-------------|
| **Architecture** | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, components, data flow |
| **Project Report** | [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) | Methodology, results, ablations |
| **PDF Report** | [docs/assets/Project_Report.pdf](docs/assets/Project_Report.pdf) | Final compiled LaTeX report |

### Key Findings

#### Metric Gains Over Baseline

Fine-tuned model improves BLEU-4 (+0.041), ROUGE-L (+0.054), CIDEr (+0.192), and brevity (Len-R drop).

| Image | Description |
|-------|-------------|
| ![Metric Lift](docs/assets/fine1.jpg) | Metric lift & brevity improvement |
| ![Core Comparison](docs/assets/fine2.jpg) | Final model vs. baseline metrics |

#### Beam Size Study

| Fluency (log-prob) | Diversity (D-1/D-2) | CIDEr | ROUGE-L |
|:------------------:|:--------------------:|:-----:|:-------:|
| ![beam1](docs/assets/beam1.png) | ![beam2](docs/assets/beam2.png) | ![beam3](docs/assets/beam3.png) | ![beam4](docs/assets/beam4.png) |

> Beam = 1 gives best overlap metrics. Diversity improves at beam = 5. Larger beams tend to collapse into generic language.

#### Cross-Attention Ablation

| Metric | With Cross-Attn | No Cross-Attn |
|--------|-----------------|---------------|
| BLEU-4 | 0.158 | 0.034 |
| METEOR | 0.483 | 0.254 |
| ROUGE-L | 0.357 | 0.223 |
| BERTScore | 0.908 | 0.874 |
| CIDEr | 0.065 | 0.002 |

> Removing cross-attention results in severe grounding loss. Captions default to memorized generic patterns.

![Cross vs No-Cross](docs/assets/cross.jpg)

#### Qualitative Examples

| Image | With Cross-Attention | Without Cross-Attention |
|-------|----------------------|-------------------------|
| ![img1](docs/assets/img1.jpg) | A yellow dog runs through grass with its tongue hanging out. | A man and a woman sit on a bench in front of a building. |
| ![img2](docs/assets/img2.jpg) | A girl in a pink hat takes a picture with a digital camera. | A man and a woman sit on a bench in front of a building. |
| ![img3](docs/assets/img3.jpg) | A person is standing in front of a golden retriever in a field. | A man and a woman sit on a bench in front of a building. |

> Add these images to `docs/assets/` (see [docs/assets/README.md](docs/assets/README.md) for the full list).

---

## Installation & Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- Docker & Docker Compose (optional)

### Local API (HuggingFace mode)

```bash
git clone https://github.com/<your-username>/Caption-It.git
cd Caption-It/final_deployment

# API
cd api && pip install -r requirements.txt
cp .env.example .env   # Optional: set INFERENCE_MODE, HF_MODEL_ID
uvicorn main:app --reload --port 8000

# Frontend (new terminal)
cd frontend && npm install && npm run dev
```

Open http://localhost:5173

### Docker (Production)

```bash
docker-compose up -d
# API: http://localhost:8000
# Frontend: http://localhost:5173
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_MODE` | `huggingface` | `huggingface` or `local` |
| `HF_MODEL_ID` | `nlpconnect/vit-gpt2-image-captioning` | Hugging Face model ID |
| `HF_API_TOKEN` | — | Optional, for gated models |
| `MODEL_PATH` | `./models/caption-it` | Path when `INFERENCE_MODE=local` |
| `MAX_SEQ_LEN` | 24 | Max caption length |
| `NUM_BEAMS` | 4 | Beam search width |

---

## Deployment

| Target | Command / Steps |
|--------|-----------------|
| **Docker Compose** | `docker-compose up -d` |
| **Hugging Face Spaces** | Copy `gradio_app/` to a new Gradio Space |
| **API only** | `docker build -t caption-it-api ./api && docker run -p 8000:8000 caption-it-api` |

---

## API Reference

| Endpoint | Method | Description |
|---------|--------|-------------|
| `/api/v1/health` | GET | Health check, inference mode |
| `/api/v1/caption` | POST | Upload image (multipart/form-data) |
| `/api/v1/caption/url` | POST | `{"url": "https://..."}` |

**Example**

```bash
curl -X POST http://localhost:8000/api/v1/caption \
  -F "file=@image.jpg"
```

Response: `{"caption": "...", "model_id": "...", "inference_time_ms": 123}`

API docs: http://localhost:8000/api/docs

---

## Research Details

### Dataset & Preprocessing

| Component | Details |
|-----------|---------|
| Dataset | [Flickr8k](https://illinois.edu/fb/sec/1713398) (8,000 images, 5 captions each) |
| Split | 6k train / 1k val / 1k test (Karpathy) |
| Preprocessing | Resize 224×224, ImageNet normalization |
| Caption cleanup | Lowercased, punctuated, NLTK tokenized, max 30 |
| Vocabulary | Freq ≥5 → 2,984 tokens incl. `<pad>`, `<unk>` |

### Training Configuration

| Component | Value |
|-----------|-------|
| Loss | Label-smoothed cross-entropy (ε=0.1) |
| Optimizer | AdamW, cosine decay, 5% warm-up |
| Batch size | 2 (effective 32 with grad accumulation) |
| Epochs | 8 (ViT unfrozen after epoch 2) |
| Precision | FP16, gradient checkpointing |
| LR | 5e-6 (encoder), 2e-5 (decoder) |

### Decoding

| Parameter | Value |
|-----------|-------|
| Method | Beam Search / Greedy |
| Beam sizes tested | 1, 3, 5, 10 |
| Length penalty | 1.6 |
| No-repeat n-gram | 5 |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| BLEU-4 | 4-gram precision overlap |
| ROUGE-L | Longest common subsequence |
| CIDEr | TF-IDF-weighted n-gram consensus |
| METEOR | Semantic/word-level match |
| BERTScore | Embedding-based similarity |

### Notebooks (Code with Outputs)

| Notebook | Description |
|----------|-------------|
| `Code With Outputs/ViT-GPT2-Finetuning.ipynb` | Main training pipeline |
| `Code With Outputs/ViT-GPT2-Crossattention-Study.ipynb` | Cross-attention ablation |
| `Code With Outputs/Beam-Size-Study.ipynb` | Beam size experiments |
| `Code With Outputs/Baseline-CNN-RNN-Models.ipynb` | ResNet+LSTM, MobileNet+GRU baselines |

---

## File Structure

```
final_deployment/
├── api/                    # FastAPI backend
│   ├── main.py             # Endpoints
│   ├── config.py           # Settings
│   ├── inference/          # HuggingFace + local captioners
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/               # Vite + TypeScript UI
│   ├── src/
│   ├── Dockerfile
│   └── nginx.conf
├── gradio_app/             # HF Spaces deployment
│   ├── app.py
│   └── requirements.txt
├── Code With Outputs/      # Jupyter notebooks (research)
│   ├── ViT-GPT2-Finetuning.ipynb
│   ├── ViT-GPT2-Crossattention-Study.ipynb
│   ├── Beam-Size-Study.ipynb
│   └── Baseline-CNN-RNN-Models.ipynb
├── docs/
│   ├── ARCHITECTURE.md     # System architecture
│   ├── PROJECT_REPORT.md   # Report summary
│   └── assets/            # PDFs, PNGs, JPGs
│       ├── archi1.pdf
│       ├── crossmodal.pdf
│       ├── Project_Report.pdf
│       ├── beam1.png … beam4.png
│       ├── cross.jpg, fine1.jpg, fine2.jpg
│       └── img1.jpg … img3.jpg
├── models/                 # Local model weights (when INFERENCE_MODE=local)
├── docker-compose.yml
├── README.md               # This file
└── requirements.txt        # Research dependencies
```

---

## Future Work

- **Larger datasets**: Flickr30k, MS COCO, SBU Captions
- **Hardware**: Multi-GPU, Cloud TPU for larger batches
- **Decoding**: Nucleus sampling, diverse beam search
- **Evaluation**: Human studies (IRB-approved)

---

## References

1. Vinyals et al., "Show and Tell," CVPR 2015
2. Karpathy & Fei-Fei, "Deep Visual-Semantic Alignments," CVPR 2015
3. Vaswani et al., "Attention is All You Need," NeurIPS 2017
4. Radford et al., "Language Models are Unsupervised Multitask Learners," OpenAI 2019
5. Wolf et al., "Transformers: State-of-the-art NLP," EMNLP Demos 2020
6. Dosovitskiy et al., "An Image is Worth 16x16 Words," ICLR 2021
7. Papineni et al., "BLEU," ACL 2002
8. Lin, "ROUGE," ACL 2004
9. Vedantam et al., "CIDEr," CVPR 2015
10. Vijayakumar et al., "Diverse Beam Search," arXiv:1606.02424

---

**Caption-It** — PhD-level image captioning with cross-modal attention. Built for research and production.
