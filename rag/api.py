from pathlib import Path
import json
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, EmailStr

from .rag_engine import SimpleRagEngine, BASE_DIR


app = FastAPI(title="b2bmasterdemo RAG API")

engine = SimpleRagEngine()
engine.build_index()


class ChatRequest(BaseModel):
  question: str


class ChatResponse(BaseModel):
  answer: str


@app.post("/api/rag-chat", response_model=ChatResponse)
def rag_chat(req: ChatRequest) -> ChatResponse:
  # İlgili dökümanları bul (log ve basit analiz için)
  docs = engine.query(req.question, top_k=3)
  answer = engine.answer_with_llm(req.question, top_k=5)

  # En ilgili dökümanın başlık/kategori bilgisini log'a yaz
  top_doc = docs[0] if docs else None
  log_entry = {
    "question": req.question,
    "answer": answer,
    "top_feature_title": top_doc["feature_title"] if top_doc else None,
    "top_category": top_doc["category"] if top_doc else None,
    "top_tags": top_doc["tags"] if top_doc else [],
    "created_at": datetime.utcnow().isoformat() + "Z",
  }

  # chat_logs.json'a ekle
  chat_log_path = BASE_DIR / "chat_logs.json"
  existing_logs: list[dict] = []
  if chat_log_path.exists():
    try:
      existing_logs = json.loads(chat_log_path.read_text(encoding="utf-8"))
    except Exception:
      existing_logs = []

  existing_logs.append(log_entry)
  chat_log_path.write_text(json.dumps(existing_logs, ensure_ascii=False, indent=2), encoding="utf-8")

  return ChatResponse(answer=answer)


# ---- Basit CRM lead kaydı ----

class LeadRequest(BaseModel):
  company_name: str
  email: EmailStr
  phone: str | None = None
  note: str | None = None


class LeadResponse(BaseModel):
  ok: bool


LEADS_PATH = BASE_DIR / "crm_leads.json"


@app.post("/api/crm-lead", response_model=LeadResponse)
def create_lead(req: LeadRequest) -> LeadResponse:
  lead = {
    "company_name": req.company_name,
    "email": req.email,
    "phone": req.phone,
    "note": req.note,
    "created_at": datetime.utcnow().isoformat() + "Z",
  }

  existing: list[dict] = []
  if LEADS_PATH.exists():
    try:
      existing = json.loads(LEADS_PATH.read_text(encoding="utf-8"))
    except Exception:
      existing = []

  existing.append(lead)
  LEADS_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

  return LeadResponse(ok=True)


@app.get("/")
def index() -> FileResponse:
  # Ana sayfa: index.html
  return FileResponse(BASE_DIR / "index.html")


@app.get("/style.css")
def style() -> FileResponse:
  # CSS dosyası
  return FileResponse(BASE_DIR / "style.css")


@app.get("/admin/leads")
def admin_leads() -> HTMLResponse:
  """
  Basit CRM ekranı: crm_leads.json içeriğini tablo olarak gösterir.
  """
  if LEADS_PATH.exists():
    try:
      leads = json.loads(LEADS_PATH.read_text(encoding="utf-8"))
    except Exception:
      leads = []
  else:
    leads = []

  rows = []
  for lead in leads:
    rows.append(
      f"<tr>"
      f"<td>{lead.get('created_at','')}</td>"
      f"<td>{lead.get('company_name','')}</td>"
      f"<td>{lead.get('email','')}</td>"
      f"<td>{lead.get('phone','')}</td>"
      f"<td>{lead.get('note','')}</td>"
      f"</tr>"
    )

  html = f"""
  <html>
  <head>
    <meta charset="utf-8" />
    <title>CRM Lead Listesi</title>
    <style>
      body {{
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        padding: 24px;
        background: #f9fafb;
      }}
      h1 {{
        margin-bottom: 16px;
        font-size: 20px;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        background: #ffffff;
      }}
      th, td {{
        border: 1px solid #e5e7eb;
        padding: 8px 10px;
        font-size: 13px;
        text-align: left;
      }}
      th {{
        background: #f3f4f6;
      }}
    </style>
  </head>
  <body>
    <h1>CRM Lead Listesi</h1>
    <table>
      <thead>
        <tr>
          <th>Oluşturma</th>
          <th>Şirket</th>
          <th>E-posta</th>
          <th>Telefon</th>
          <th>Not</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </body>
  </html>
  """
  return HTMLResponse(content=html)


@app.get("/admin/chats")
def admin_chats() -> HTMLResponse:
  """
  Chatbot üzerinden gelen soruların ve en çok hangi konulara denk geldiklerinin
  basit bir görünümü ve kategori bazlı küçük bir özet.
  """
  chat_log_path = BASE_DIR / "chat_logs.json"
  if chat_log_path.exists():
    try:
      logs = json.loads(chat_log_path.read_text(encoding="utf-8"))
    except Exception:
      logs = []
  else:
    logs = []

  # Kategori bazlı sayım
  category_counts: dict[str, int] = {}
  for log in logs:
    cat = log.get("top_category") or "Bilinmiyor"
    category_counts[cat] = category_counts.get(cat, 0) + 1

  cat_data_json = json.dumps(category_counts, ensure_ascii=False)

  # Chat log tablosu (son 200 kayıt)
  rows = []
  for log in reversed(logs[-200:]):
    rows.append(
      "<tr>"
      f"<td>{log.get('created_at','')}</td>"
      f"<td>{(log.get('question','') or '')[:120]}</td>"
      f"<td>{log.get('top_feature_title','')}</td>"
      f"<td>{log.get('top_category','')}</td>"
      "</tr>"
    )

  html = f"""
  <html>
  <head>
    <meta charset="utf-8" />
    <title>Chat Logları · b2bmasterdemo</title>
    <style>
      body {{
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        padding: 24px;
        background: #f9fafb;
      }}
      h1 {{
        margin-bottom: 4px;
        font-size: 20px;
      }}
      h2 {{
        margin: 16px 0 8px;
        font-size: 16px;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        background: #ffffff;
      }}
      th, td {{
        border: 1px solid #e5e7eb;
        padding: 8px 10px;
        font-size: 13px;
        text-align: left;
        vertical-align: top;
      }}
      th {{
        background: #f3f4f6;
      }}
    </style>
  </head>
  <body>
    <h1>Chatbot Soruları</h1>

    <h2>Kategori Özeti</h2>
    <div style="display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap;">
      <canvas id="catPie" width="260" height="260" style="background:#ffffff;border-radius:12px;border:1px solid #e5e7eb;"></canvas>
      <div id="catLegend" style="font-size:13px;color:#374151;"></div>
    </div>
    <script>
      (function() {{
        var data = {cat_data_json};
        var canvas = document.getElementById('catPie');
        var legend = document.getElementById('catLegend');
        if (!canvas || !legend) return;
        var ctx = canvas.getContext('2d');
        var entries = Object.entries(data);
        if (!entries.length) {{
          legend.innerHTML = "<div style='font-size:13px;color:#6b7280;'>Henüz kayıt yok.</div>";
          return;
        }}
        var total = entries.reduce(function (s, e) {{ return s + e[1]; }}, 0) || 1;
        var colors = ['#2563eb','#16a34a','#f97316','#dc2626','#7c3aed','#0891b2','#4b5563'];
        var start = -Math.PI / 2;
        entries.forEach(function (entry, idx) {{
          var label = entry[0];
          var value = entry[1];
          var angle = (value / total) * Math.PI * 2;
          ctx.beginPath();
          ctx.moveTo(canvas.width / 2, canvas.height / 2);
          ctx.arc(
            canvas.width / 2,
            canvas.height / 2,
            Math.min(canvas.width, canvas.height) / 2 - 8,
            start,
            start + angle
          );
          ctx.closePath();
          ctx.fillStyle = colors[idx % colors.length];
          ctx.fill();
          start += angle;
        }});

        var legendHtml = "";
        entries.forEach(function (entry, idx) {{
          var label = entry[0];
          var value = entry[1];
          var percent = Math.round((value / total) * 100);
          legendHtml += "<div style='display:flex;align-items:center;margin-bottom:4px;'>";
          legendHtml += "<span style='display:inline-block;width:10px;height:10px;border-radius:2px;background:" +
                        colors[idx % colors.length] +
                        ";margin-right:6px;'></span>";
          legendHtml += "<span>" + label + " – " + value + " soru (%" + percent + ")</span>";
          legendHtml += "</div>";
        }});
        legend.innerHTML = legendHtml;
      }})();
    </script>

    <h2>Son Sorular</h2>
    <table>
      <thead>
        <tr>
          <th>Zaman</th>
          <th>Soru</th>
          <th>Eşleşen Özellik</th>
          <th>Kategori</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </body>
  </html>
  """

  return HTMLResponse(content=html)


def create_app() -> FastAPI:
  # Uvicorn için referans amaçlı factory
  return app

