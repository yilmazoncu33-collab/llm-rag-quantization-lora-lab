from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["loaded_docs"] == 4


def test_grounded_technical_question():
    response = client.post("/ask", json={"question": "C-Bant veri linki menzili kaç km?"})
    assert response.status_code == 200
    data = response.json()
    assert data["is_grounded"] is True
    assert data["source_id"] == "DOC-02-HABERLESME"
    assert "150 km" in data["answer"]


def test_hallucination_guardrail():
    response = client.post("/ask", json={"question": "Denizaltı sonar frekansı kaç kHz?"})
    assert response.status_code == 200
    data = response.json()
    assert data["is_grounded"] is False
    assert data["source_id"] == "NONE"
    assert "YETERSİZ BİLGİ" in data["answer"]
