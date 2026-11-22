from dataclasses import dataclass
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class EngineResponse:
    answer: str
    source_id: str
    confidence: float
    is_grounded: bool


class UAVKnowledgeEngine:
    """İHA teknik dokümanları üzerinde deterministik RAG ve halüsinasyon kontrolü yapan çekirdek motor."""

    def __init__(self, documents: List[Dict[str, str]], confidence_threshold: float = 0.20):
        if not documents:
            raise ValueError("Doküman listesi boş olamaz.")
        self.documents = documents
        self.confidence_threshold = confidence_threshold
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.doc_vectors = self.vectorizer.fit_transform([doc["content"] for doc in documents])

    def query(self, question: str) -> EngineResponse:
        clean_q = question.strip()
        if not clean_q:
            raise ValueError("Soru metni boş olamaz.")

        q_vec = self.vectorizer.transform([clean_q])
        similarities = cosine_similarity(q_vec, self.doc_vectors).flatten()

        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])
        best_doc = self.documents[best_idx]

        # Güven eşiği altındaysa uydurma (halüsinasyon) yerine güvenli ret döndür
        if best_score < self.confidence_threshold:
            return EngineResponse(
                answer="YETERSİZ BİLGİ: Teknik dokümanlarda bu soruya ait doğrulanmış veri bulunamadı.",
                source_id="NONE",
                confidence=round(best_score, 4),
                is_grounded=False
            )

        return EngineResponse(
            answer=best_doc["content"],
            source_id=best_doc["id"],
            confidence=round(best_score, 4),
            is_grounded=True
        )


DEFAULT_UAV_DOCS = [
    {
        "id": "DOC-01-GENEL",
        "content": "TB-X İHA azami kalkış ağırlığı (MTOW) 750 kg olup faydalı yük kapasitesi 150 kg'dır."
    },
    {
        "id": "DOC-02-HABERLESME",
        "content": "C-Bant görüş hattı (LOS) veri linki menzili açık havada 150 km'dir."
    },
    {
        "id": "DOC-03-MOTOR",
        "content": "Motor egzoz gazı sıcaklığı (EGT) 780 dereceyi aştığında sistem otomatik güç keser."
    },
    {
        "id": "DOC-04-AVIYONIK",
        "content": "INS/GPS hibrit navigasyon modülü saniyede 100 Hz frekansında konum günceller."
    }
]
