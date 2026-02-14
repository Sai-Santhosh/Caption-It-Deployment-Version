# Caption-It: Project Report Summary

> **Academic Report** — Vision-Language Image Captioning with Cross-Modal Attention

---

## Executive Summary

Caption-It is an end-to-end image captioning system that leverages a frozen Vision Transformer (ViT) encoder and a GPT-2 decoder with full cross-modal attention. Fine-tuned on the Flickr8k dataset with a custom 2-layer MLP bridge and progressive unfreezing, the system **outperforms traditional CNN–RNN baselines** across BLEU-4, ROUGE-L, and CIDEr. Exhaustive ablation studies validate the **critical role of visual grounding via cross-attention**.

---

## Key Findings

### 1. Baseline Comparison

| Model               | BLEU-4 | ROUGE-L | CIDEr | Notes                    |
|---------------------|--------|---------|-------|--------------------------|
| ResNet50 + LSTM     | 0.117  | 0.303   | —     | Frozen ResNet, 512-d LSTM |
| MobileNetV3 + GRU   | 0.089  | 0.271   | —     | Lighter baseline          |
| **ViT-GPT2 (Ours)** | **0.158** | **0.357** | **0.065** | Cross-attention, MLP bridge |

### 2. Cross-Attention Ablation

Removing cross-attention results in **severe grounding loss**. The model defaults to memorized generic patterns:

| Metric    | With Cross-Attn | No Cross-Attn |
|-----------|-----------------|---------------|
| BLEU-4    | 0.158           | 0.034         |
| METEOR    | 0.483           | 0.254         |
| ROUGE-L   | 0.357           | 0.223         |
| BERTScore | 0.908           | 0.874         |
| CIDEr     | 0.065           | 0.002         |

### 3. Beam Size Study

- **Beam = 1**: Best overlap metrics (BLEU, ROUGE-L, CIDEr)
- **Beam = 5**: Optimal diversity (D-1, D-2)
- **Beam ≥ 10**: Captions collapse into generic language

### 4. Fine-Tuning Gains

- BLEU-4: **+0.041**
- ROUGE-L: **+0.054**
- CIDEr: **+0.192**
- Brevity (Len-R) improved (less repetition)

---

## Methodology

1. **Data**: Flickr8k (8,091 images, 5 captions each), Karpathy split (6k/1k/1k)
2. **Preprocessing**: 224×224, ImageNet normalization, NLTK tokenization, vocab ≥5
3. **Training**: Label smoothing (ε=0.1), AdamW, cosine decay, 5% warm-up
4. **Decoding**: Beam search, length penalty 1.6, no-repeat 5-gram

---

## References

See main README for full citation list.

---

*Full LaTeX report: `docs/Project_Report.pdf` (add when compiled)*
