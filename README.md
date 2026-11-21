# UAV Domain Adaptation & Optimization with LLMs (LoRA, Quantization & RAG)

This repository explores practical methods for adapting and optimizing a lightweight Large Language Model (`Qwen/Qwen2.5-1.5B-Instruct`) for Unmanned Aerial Vehicle (UAV) domain tasks, comparing parameter-efficient fine-tuning, 4-bit quantization, and retrieval-based workflows.

## Core Components
* **LoRA Fine-Tuning (`PEFT`):** Adapting `Qwen2.5-1.5B-Instruct` (`r=8`, `lora_alpha=16`) on domain-specific UAV Q&A examples while training only a small fraction of parameters (~1.09M trainable parameters out of ~1.54B, ~0.07%).
* **4-Bit Quantization (`bitsandbytes` / `NF4`):** Benchmarking `FP16` vs. `4-bit NF4` (`float16` compute dtype) on GPU to evaluate VRAM savings and inference speed trade-offs.
* **RAG & Embeddings (`sentence-transformers`):** Combining dense vector retrieval with LLM generation for context-grounded responses.

## Benchmark Highlights (T4 GPU)
| Configuration | VRAM Usage | Latency | Trainable Parameters |
| :--- | :---: | :---: | :---: |
| **FP16 Baseline** | 2.88 GB | 2.05 s | 100% (~1.54B) |
| **4-Bit NF4** | 1.14 GB | 1.19 s | - |
| **LoRA Adapter** | ~2.90 GB | ~2.10 s | ~0.07% (~1.09M) |

## Key Takeaways & Limitations
* **Memory vs. Speed:** 4-bit NF4 significantly lowers VRAM usage and speeds up inference, though smaller models can show increased hallucination when domain knowledge is not grounded via fine-tuning or RAG.
* **Fine-Tuning Sensitivity:** Small datasets and unmasked prompt tokens slow down convergence—using chat templates, masking prompt loss (`-100`), and expanding training data/epochs improve response reliability.

## Setup
```bash
pip install -q torch transformers accelerate peft datasets bitsandbytes sentence-transformers
