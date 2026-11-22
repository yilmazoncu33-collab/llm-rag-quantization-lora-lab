import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from engine import UAVKnowledgeEngine, DEFAULT_UAV_DOCS

app = FastAPI(
    title="İHA Teknik Doküman Sorgulama API",
    description="Üretime hazır, halüsinasyon kontrollü NLP/RAG mikroservisi.",
    version="1.0.0"
)

engine = UAVKnowledgeEngine(documents=DEFAULT_UAV_DOCS, confidence_threshold=0.20)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, description="Operatör sorusu")


class AnswerResponse(BaseModel):
    question: str
    answer: str
    source_id: str
    confidence: float
    is_grounded: bool
    latency_ms: float


@app.get("/health")
def health_check():
    return {"status": "healthy", "loaded_docs": len(engine.documents)}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest):
    t0 = time.perf_counter()
    try:
        result = engine.query(payload.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    latency_ms = round((time.perf_counter() - t0) * 1000, 3)

    return AnswerResponse(
        question=payload.question,
        answer=result.answer,
        source_id=result.source_id,
        confidence=result.confidence,
        is_grounded=result.is_grounded,
        latency_ms=latency_ms
    )
