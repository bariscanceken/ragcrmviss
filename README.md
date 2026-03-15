# b2bmasterdemo – RAG + Chatbot + Basit CRM

Bu proje, b2bmasterdemo için:

- Sade bir **B2B landing page** (`index.html` + `style.css`)
- RAG + LLM tabanlı **chatbot**
- Form ve chatbot üzerinden gelen talepleri tutan küçük bir **CRM**
- Chat sorularını kaydedip basit bir **analiz/görselleştirme** ekranı

sağlar.

---

## Mimari Genel Bakış

### Frontend

- `index.html`
  - Hero, hizmetler, “Demo Talep Formu”
  - Sağ altta RAG + LLM ile konuşan sohbet balonu
- `style.css`
  - Tüm sayfanın modern ve sade tasarımı
  - Chat penceresi ve formlar dahil

### RAG Katmanı

- `db.json`
  - b2bmasterdemo'nun misyon, vizyon, modüller, KVKK, use-case’ler vb. bilgi tabanı
  - Embedding üretimi ve sorgu için tek kaynak
- `rag/rag_engine.py`
  - `SimpleRagEngine` sınıfı:
    - `load_documents()` → `db.json`’u okur (`RagDocument` listesi)
    - `build_index()` → `SentenceTransformer` ile embedding üretir
    - `query(question, top_k)` → cosine similarity ile en alakalı kayıtları bulur
    - `answer_with_llm(question, top_k, model)` → RAG sonuçlarını context olarak OpenAI LLM’e verip cevap üretir
  - Cosine similarity, yalnızca **numpy** ile hesaplanır, `scikit-learn` gerekmez.

### Backend (FastAPI API)

`rag/api.py` FastAPI uygulamasını içerir:

- Uygulama başlarken:

  ```python
  engine = SimpleRagEngine()
  engine.build_index()
  ```

- Endpoint’ler:
  - `GET /`
    - `index.html` döner (ana sayfa)
  - `GET /style.css`
    - CSS dosyasını döner
  - `POST /api/rag-chat`
    - Body: `{"question": "kullanıcının sorusu"}`
    - RAG + LLM cevabı döner: `{"answer": "..."}`
    - Ek olarak `chat_logs.json` içine log yazar:
      - `question`, `answer`, `top_feature_title`, `top_category`, `top_tags`, `created_at`
  - `POST /api/crm-lead`
    - Body:

      ```json
      {
        "company_name": "Örn: ABC Dış Ticaret A.Ş.",
        "email": "ornek@firma.com",
        "phone": "+90...",        // opsiyonel
        "note": "kısa not"        // opsiyonel
      }
      ```

    - `crm_leads.json` içine lead ekler.
  - `GET /admin/leads`
    - Tüm lead’leri tablo halinde gösteren basit CRM ekranı
  - `GET /admin/chats`
    - `chat_logs.json` üzerinden:
      - Kategori sayımlarına göre **pasta grafiği (pie chart)** (vanilla JS + canvas)
      - Altında son 200 soruyu tablo halinde gösterir:
        - Zaman, soru, eşleşen özellik başlığı, kategori

### Veri Dosyaları

- `db.json` → RAG bilgi tabanı
- `crm_leads.json` → Lead kayıtları
- `chat_logs.json` → Chat soruları + cevaplar + RAG eşleşmesi

---

## Akışlar

### Chatbot Akışı

1. Kullanıcı sağ alttaki sohbet balonunu açar.
2. Frontend JS:
   - `fetch("/api/rag-chat", { body: { question } })` çağrısı yapar.
3. Backend:
   - `engine.query(question, top_k=3)` ile en alakalı kayıtları bulur.
   - `engine.answer_with_llm(...)` ile OpenAI modeli üzerinden nihai cevabı üretir.
   - Soru + cevap + kategori özeti `chat_logs.json` içine kaydedilir.
4. Frontend:
   - Soruyu “user”, cevabı “assistant” satırı olarak chat penceresinde gösterir.

### CRM Akışı

1. Kullanıcı:
   - Ana sayfadaki **Demo Talep Formu**nu doldurur veya
   - Chat penceresindeki mini **“Görüşme talebi oluştur”** formunu kullanır.
2. JS:
   - `POST /api/crm-lead` ile sunucuya istek atar.
3. Backend:
   - Lead objesini `crm_leads.json` içine ekler.
4. Admin:
   - `GET /admin/leads` ile tüm lead’leri görür.
   - `GET /admin/chats` ile chatbot sorularını ve kategori pastasını görür.

---

## Kurulum

### 1. Ortam Hazırlığı

Proje klasörüne geç:

```powershell
cd "C:\Users\baris\OneDrive\Masaüstü\ragcrmviss"
```

İsteğe bağlı sanal ortam:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Bağımlılıkların Kurulması

`rag/requirements.txt` içeriği:

```txt
sentence-transformers==3.3.1
numpy>=2.1.0
openai==1.58.1
python-dotenv==1.0.1
fastapi==0.115.6
uvicorn==0.32.1
email-validator==2.2.0
```

Kurulum:

```powershell
pip install --upgrade pip
pip install -r rag\requirements.txt
```

### 3. OpenAI API Anahtarı

Proje kökünde `.env` dosyası oluştur:

```env
OPENAI_API_KEY=buraya_senin_key
```

`rag/rag_engine.py` içindeki `_ensure_llm_client()` `.env` dosyasını okuyarak bu anahtarı kullanır.

---

## Çalıştırma

FastAPI + RAG sunucusunu başlat:

```powershell
cd "C:\Users\baris\OneDrive\Masaüstü\ragcrmviss"
python -m uvicorn rag.api:app --reload
```

Terminalde:

```text
Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### 1. Ana Uygulama

Tarayıcıda:

```text
http://127.0.0.1:8000/
```

- Landing page + CSS ile görünür.
- Sağ altta chatbot:
  - Örneğin “Aidat sistemi ne yapıyor?” diye sorarak test edebilirsin.
- Demo talep formunu doldurarak lead bırakabilirsin.

### 2. API Dokümantasyonu

```text
http://127.0.0.1:8000/docs
```

Buradan:

- `POST /api/rag-chat`
- `POST /api/crm-lead`
- `GET /admin/leads`
- `GET /admin/chats`

endpoint’lerini görebilir ve test edebilirsin.

### 3. Admin Ekranları

- Lead listesi:

  ```text
  http://127.0.0.1:8000/admin/leads
  ```

- Chat log + kategori pasta grafiği:

  ```text
  http://127.0.0.1:8000/admin/chats
  ```

---

## Geliştirme Önerileri

- `db.json` içeriğini zenginleştirerek RAG cevaplarını daha kapsamlı hale getirmek.
- `system_prompt` metnini güncelleyerek cevapların tonunu (daha resmi/samimi, daha kısa/uzun) ayarlamak.
- `crm_leads.json` ve `chat_logs.json` yerine gerçek bir veritabanına geçip (PostgreSQL vb.), BI aracı ile raporlama yapmak.
- Lead kaydı yapılırken o anda açık olan son soruları `note` içine otomatik eklemek.

