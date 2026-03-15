import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
# Tüm RAG verilerini tuttuğumuz JSON dosyası
DATA_PATH = BASE_DIR / "db.json"


@dataclass
class RagDocument:
    id: str
    feature_title: str
    category: str
    content: str
    technical_details: str
    tags: List[str]
    benefit_score: float

    @property
    def full_text(self) -> str:
        parts = [
            f"Başlık: {self.feature_title}",
            f"Kategori: {self.category}",
            self.content or "",
            f"Teknik Detaylar: {self.technical_details}" if self.technical_details else "",
            f"Etiketler: {', '.join(self.tags)}" if self.tags else "",
        ]
        return "\n".join(p for p in parts if p)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    sklearn'e ihtiyaç duymadan basit cosine similarity hesabı.
    a: (1, d), b: (n, d) -> (1, n)
    """
    # Güvenlik için kopya alıyoruz ki orijinal embedding'leri bozmayalım.
    a_norm = a.astype(np.float32, copy=True)
    b_norm = b.astype(np.float32, copy=True)

    a_norm /= np.linalg.norm(a_norm, axis=1, keepdims=True) + 1e-8
    b_norm /= np.linalg.norm(b_norm, axis=1, keepdims=True) + 1e-8
    return a_norm @ b_norm.T


class SimpleRagEngine:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model: SentenceTransformer | None = None
        self.documents: List[RagDocument] = []
        self.embeddings: np.ndarray | None = None
        # LLM istemcisi (isteğe bağlı)
        self.llm_client: OpenAI | None = None

    def load_documents(self, path: Path = DATA_PATH) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        docs: List[RagDocument] = []
        for item in raw:
            docs.append(
                RagDocument(
                    id=item.get("id", ""),
                    feature_title=item.get("feature_title", ""),
                    category=item.get("category", ""),
                    content=item.get("content", ""),
                    technical_details=item.get("technical_details", ""),
                    tags=item.get("tags", []) or [],
                    benefit_score=float(item.get("benefit_score", 0.0)),
                )
            )
        self.documents = docs

    def _ensure_model(self) -> None:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def build_index(self) -> None:
        if not self.documents:
            self.load_documents()
        self._ensure_model()
        texts = [doc.full_text for doc in self.documents]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def query(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.embeddings is None or not self.documents:
            self.build_index()
        assert self.embeddings is not None
        self._ensure_model()

        q_vec = self.model.encode([question], convert_to_numpy=True)
        sims = _cosine_similarity(q_vec, self.embeddings)[0]
        top_idx = sims.argsort()[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        for idx in top_idx:
            doc = self.documents[int(idx)]
            results.append(
                {
                    "id": doc.id,
                    "feature_title": doc.feature_title,
                    "category": doc.category,
                    "score": float(sims[idx]),
                    "content": doc.content,
                    "technical_details": doc.technical_details,
                    "tags": doc.tags,
                }
            )
        return results

    # -------- LLM ile cevap üretimi --------

    def _ensure_llm_client(self) -> None:
        """
        OPENAI_API_KEY ortam değişkenini .env'den yükleyip OpenAI client oluşturur.
        """
        if self.llm_client is not None:
            return

        # .env varsa yükle
        env_path = BASE_DIR / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY bulunamadı. Lütfen proje kökünde .env içine OPENAI_API_KEY=... ekle.")

        # Ortam değişkeninde olduğundan burada ayrıca parametre vermeye gerek yok
        self.llm_client = OpenAI()

    def answer_with_llm(
        self,
        question: str,
        top_k: int = 5,
        model: str = "gpt-4o-mini",
    ) -> str:
        """
        - Önce RAG ile en ilgili kayıtları bulur.
        - Sonra bu context'i kullanarak LLM'den nihai cevabı üretir.
        """
        docs = self.query(question, top_k=top_k)
        self._ensure_llm_client()
        assert self.llm_client is not None

        context_chunks = []
        for d in docs:
            chunk = (
                f"ID: {d['id']}\n"
                f"Başlık: {d['feature_title']}\n"
                f"Kategori: {d['category']}\n"
                f"Etiketler: {', '.join(d['tags'])}\n"
                f"İçerik: {d['content']}\n"
                f"Teknik Detaylar: {d['technical_details']}\n"
            )
            context_chunks.append(chunk)

        context_text = "\n\n---\n\n".join(context_chunks)

        system_prompt = (
            "Sen b2bmasterdemo için çalışan bir yardımcı asistansın. "
            "Aşağıda sana verilen bağlam (context) yalnızca bu ürünün özellikleri, "
            "kullanım senaryoları, güvenlik ve teknik detayları ile ilgilidir. "
            "Kullanıcı sorularına sadece bu bağlamdan yola çıkarak, Türkçe ve sade bir dille cevap ver. "
            "Bağlamda olmayan finansal rakamlar veya hukuki detaylar için 'bu bilgi dokümanda yok' de."
        )

        user_prompt = (
            f"Kullanıcı sorusu:\n{question}\n\n"
            f"İlgili bağlam:\n{context_text}\n\n"
            "Lütfen yukarıdaki bağlama dayanarak net, kısa ve anlaşılır bir cevap üret."
        )

        completion = self.llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return completion.choices[0].message.content.strip()


def demo_cli() -> None:
    engine = SimpleRagEngine()
    engine.build_index()
    print("b2bmasterdemo RAG + LLM (yalnızca RAG için: 'r', LLM cevabı için: 'l', çıkmak için boş geç):")
    while True:
        try:
            q = input("\nSoru: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        mode = input("Sadece RAG sonuçları için 'r', LLM cevabı için 'l' girin [l]: ").strip().lower() or "l"

        if mode == "r":
            answers = engine.query(q, top_k=3)
            for i, ans in enumerate(answers, start=1):
                print(f"\n#{i} [{ans['score']:.3f}] {ans['feature_title']} ({ans['category']})")
                print(ans["content"][:600], "..." if len(ans["content"]) > 600 else "")
        else:
            try:
                answer = engine.answer_with_llm(q, top_k=5)
                print("\nLLM Cevabı:\n")
                print(answer)
            except Exception as exc:
                print(f"\nLLM çağrısında hata: {exc}")


if __name__ == "__main__":
    demo_cli()

