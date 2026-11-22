# Applied LLM Systems & Production NLP Engineering

An end-to-end, reproducible engineering portfolio covering the complete lifecycle of modern Large Language Model (LLM) and Natural Language Processing (NLP) systems—from **Retrieval-Augmented Generation (RAG)**, **PEFT/LoRA fine-tuning**, and **Direct Preference Optimization (DPO)** to **model compression (Quantization, Pruning, Knowledge Distillation)**, **LLMOps evaluation pipelines**, and **containerized FastAPI deployment**.

Each module is designed with isolated, reproducible experiments, empirical benchmarks, and strict hallucination-control guardrails tailored for mission-critical technical and aerospace/UAV operational domains.

---

## Repository Structure

```text
applied-llm-systems-engineering/
├── 01_RAG_Pipeline.ipynb
├── 02_Quantization_Optimization.ipynb
├── 03_LoRA_SFT_Alignment.ipynb
├── 04_End_to_End_Hybrid_Pipeline.ipynb
├── 05_DPO_Preference_Alignment.ipynb
├── 06_Pruning_ve_Distillation_Optimizasyon.ipynb
├── 07_LLMOps_Evaluation_AB_Test.ipynb
├── 08_Production_NLP_Service.ipynb
├── 08_Production_NLP_Service/
│   ├── engine.py            # Modular TF-IDF/Cosine RAG & hallucination guardrail engine
│   ├── app.py               # FastAPI REST service (/health, /ask) with Pydantic schemas
│   ├── test_app.py          # Automated unit & guardrail tests via pytest
│   ├── requirements.txt     # Pinned production dependencies
│   └── Dockerfile           # Container image build with automated pre-deploy pytest check
└── README.md


Modules & Empirical Benchmarks
1. Retrieval-Augmented Generation (01_RAG_Pipeline.ipynb)
Objective: Ground LLM responses in domain-specific technical documentation to eliminate parametric hallucination.
Implementation: Document chunking, dense vector embeddings, semantic similarity search, and strict context-injected prompt construction.

2. Memory & Latency Optimization via Quantization (02_Quantization_Optimization.ipynb)
Objective: Enable large Transformer models to run under strict edge/embedded VRAM constraints.
Implementation: Compared FP16 vs. 8-bit and 4-bit (NF4 with double quantization via bitsandbytes), measuring VRAM footprint reduction and inference latency trade-offs.

3. Parameter-Efficient Fine-Tuning (03_LoRA_SFT_Alignment.ipynb)
Objective: Adapt a pre-trained causal LLM to domain-specific technical terminology and structured response formats without full-parameter retraining.
Implementation: Supervised Fine-Tuning (SFT) using LoRA (Low-Rank Adaptation) on attention projection layers, updating <1% of total trainable parameters.

4. End-to-End Hybrid Pipeline (04_End_to_End_Hybrid_Pipeline.ipynb)
Objective: Combine dynamic external knowledge retrieval with domain-adapted generation weights.
Implementation: Integrated the vector retrieval layer with a LoRA fine-tuned model to handle multi-document technical queries with verifiable source attribution.

5. Preference Alignment with DPO (05_DPO_Preference_Alignment.ipynb)
Objective: Enforce strict context-faithfulness and refusal behavior when queries fall outside retrieved operational manuals.
Implementation: Trained using Direct Preference Optimization (DPO) on (prompt, chosen, rejected) triplets, penalizing speculative parametric answers and rewarding explicit abstention.

6. Model Compression: Pruning & Knowledge Distillation (06_Pruning_ve_Distillation_Optimizasyon.ipynb)Objective: Compress a 12-layer Turkish BERT classifier (dbmdz/bert-base-turkish-cased) for low-latency edge deployment without sacrificing predictive quality.Isolated Experiment Results:One-Shot L1 Unstructured Pruning (30% Sparsity): Zeroed out 25,658,030 weights across 74 Linear layers (110.6M $\rightarrow$ 84.9M active parameters), demonstrating the expected accuracy drop (83.3% $\rightarrow$ 50.0%) when applied without iterative recovery fine-tuning.Knowledge Distillation (T = 4.0, α = 0.7): Distilled soft logits and ground-truth signals from an undamaged 12-layer Teacher into a 2-layer Student architecture:
Metric,Teacher Model (12 Layers),Student Model (2 Layers),Impact
Total Parameters,"110,619,651","39,740,931",-64.1% parameters
Memory Footprint,422.0 MB,151.6 MB,-64.1% size
Inference Latency,12.15 ms / query,2.79 ms / query,4.35x faster
Test Accuracy,66.7%,100.0%,Regularization & soft-target gain

7. LLMOps, Automated Evaluation & A/B Testing (07_LLMOps_Evaluation_AB_Test.ipynb)
Objective: Establish a reproducible evaluation harness comparing system versions across accuracy, hallucination rejection, latency, and token cost.
Implementation: Evaluated v1.0-baseline (temp=0.7) against v2.0-strict-ops (temp=0.1, strict grounding instructions) on a 10-question Golden Evaluation Dataset (including 3 out-of-context hallucination traps) with automated JSON experiment logging (llmops_experiment_logs.json):
System Version,Accuracy (10 Qs),Hallucination Traps Passed,Avg. Latency,Avg. Output Tokens
v1.0-baseline,70.0% (7/10),0 / 3,6.398 s,67.3 tokens
v2.0-strict-ops,100.0% (10/10),3 / 3,0.288 s,6.3 tokens

8. Production Microservice, Testing & Containerization (08_Production_NLP_Service/)
Objective: Transition experimental NLP/RAG logic into a modular, tested, and containerized REST microservice.
Architecture:
engine.py: Decoupled, type-annotated UAVKnowledgeEngine enforcing a 0.20 cosine-similarity confidence threshold to prevent ungrounded responses.
app.py: FastAPI application exposing GET /health and POST /ask endpoints with strict Pydantic request/response validation and millisecond latency telemetry.
test_app.py: Automated pytest suite verifying service health, grounded technical retrieval, and out-of-domain hallucination guardrails (3 passed).
Dockerfile: Reproducible python:3.11-slim container build that executes pytest -v during image creation before exposing port 8000.

Quickstart: Running the Production Service
Local Execution & Unit Testing
Bash
cd 08_Production_NLP_Service
pip install -r requirements.txt

# Run automated unit and guardrail tests
pytest -v

# Start the FastAPI server
uvicorn app:app --host 0.0.0.0 --port 8000
Docker Deployment
Bash
cd 08_Production_NLP_Service
docker build -t uav-nlp-service:1.0.0 .
docker run -p 8000:8000 uav-nlp-service:1.0.0
Example API Request (POST /ask)
Bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "C-Bant veri linki menzili kaç km?"}'
JSON
{
  "question": "C-Bant veri linki menzili kaç km?",
  "answer": "C-Bant görüş hattı (LOS) veri linki menzili açık havada 150 km'dir.",
  "source_id": "DOC-02-HABERLESME",
  "confidence": 0.4812,
  "is_grounded": true,
  "latency_ms": 2.41
}
Tech Stack

Deep Learning & LLM Frameworks: PyTorch, Hugging Face transformers, peft (LoRA), trl (DPO), bitsandbytes, accelerate
Information Retrieval & Evaluation: scikit-learn (TF-IDF / Cosine Similarity), Dense Embeddings, pandas
Backend, Testing & DevOps: FastAPI, Uvicorn, Pydantic, pytest, httpx, Docker
