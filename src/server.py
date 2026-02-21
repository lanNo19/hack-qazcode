"""
Medical Diagnosis Server
Replaces mock_server.py with real RAG + LLM pipeline.

Usage:
    uv run uvicorn src.server:app --host 0.0.0.0 --port 8080

Docker:
    docker build -t submission .
    docker run -p 8080:8080 submission

Endpoint: POST /diagnose  {"symptoms": "..."}
"""

import json
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI

# --- Import your retriever ---
import sys
sys.path.insert(0, os.path.dirname(__file__))
from retrieval_and_reranking import MedicalRetriever

from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------------
# Configuration — set via environment variables or edit here
# ---------------------------------------------------------------------------
HUB_URL   = os.environ.get("HUB_URL", "YOUR_HUB_URL")
API_KEY   = os.environ.get("API_KEY", "YOUR_API_KEY")
LLM_MODEL = "oss-120b"

CORPUS_FILE = os.environ.get("CORPUS_FILE", "processed_corpus.json")
EMBED_FILE  = os.environ.get("EMBED_FILE",  "embeddings.npz")
SPARSE_FILE = os.environ.get("SPARSE_FILE", "sparse_weights.json")
BM25_FILE   = os.environ.get("BM25_FILE",   "bm25_index.pkl")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
Ты — система клинической поддержки принятия решений, работающая на основе \
официальных клинических протоколов Министерства здравоохранения Республики Казахстан.

Тебе будут предоставлены:
1. Симптомы пациента
2. Фрагменты релевантных клинических протоколов с соответствующими кодами МКБ-10

Твоя задача — выбрать наиболее вероятные диагнозы ИСКЛЮЧИТЕЛЬНО из предоставленных протоколов.

ВАЖНЫЕ ПРАВИЛА:
- Используй ТОЛЬКО коды МКБ-10 из списка AVAILABLE_ICD_CODES, приведённого ниже
- НЕ придумывай коды МКБ-10, которых нет в списке
- Выбери 3–5 наиболее подходящих диагнозов, ранжированных по вероятности
- Для каждого диагноза дай краткое клиническое обоснование на основе симптомов

Отвечай ТОЛЬКО валидным JSON в следующем формате (без markdown, без пояснений):
{
  "diagnoses": [
    {
      "rank": 1,
      "icd10_code": "КОД_МКБ",
      "diagnosis": "Название диагноза",
      "explanation": "Краткое клиническое обоснование"
    }
  ]
}
"""

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
retriever: MedicalRetriever = None
llm_client: OpenAI = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, llm_client

    print("\n🏥 Medical Diagnosis Server")
    print("=" * 40)
    print("Loading retriever models and indexes...")

    retriever = MedicalRetriever(CORPUS_FILE, EMBED_FILE, SPARSE_FILE, BM25_FILE)

    llm_client = OpenAI(base_url=HUB_URL, api_key=API_KEY)

    print("✓ Ready. Endpoint: POST /diagnose")
    print("=" * 40 + "\n")
    yield


app = FastAPI(title="Medical Diagnosis Server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / Response models (matches evaluator exactly)
# ---------------------------------------------------------------------------
class DiagnoseRequest(BaseModel):
    symptoms: str = ""


class Diagnosis(BaseModel):
    rank: int
    icd10_code: str
    diagnosis: str
    explanation: str


class DiagnoseResponse(BaseModel):
    diagnoses: list[Diagnosis]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def build_context(retrieved_chunks: list[dict]) -> tuple[str, list[str]]:
    """
    Format retrieved chunks into LLM context string and collect all available ICD codes.
    Returns (context_string, flat_list_of_icd_codes).
    """
    context_parts = []
    all_icd_codes = []

    for i, chunk in enumerate(retrieved_chunks, 1):
        icd_codes = chunk["metadata"].get("icd_codes", [])
        all_icd_codes.extend(icd_codes)
        icd_str = ", ".join(icd_codes) if icd_codes else "не указан"
        context_parts.append(
            f"--- Протокол {i} (МКБ-10: {icd_str}) ---\n{chunk['content']}"
        )

    context_str = "\n\n".join(context_parts)
    # Deduplicate while preserving order
    seen = set()
    unique_icd = [c for c in all_icd_codes if not (c in seen or seen.add(c))]
    return context_str, unique_icd


def call_llm(symptoms: str, context: str, available_icd_codes: list[str]) -> list[dict]:
    """Call the GPT-OSS LLM and parse the structured JSON response."""
    icd_list_str = ", ".join(available_icd_codes)

    user_message = (
        f"AVAILABLE_ICD_CODES: {icd_list_str}\n\n"
        f"КЛИНИЧЕСКИЕ ПРОТОКОЛЫ:\n{context}\n\n"
        f"СИМПТОМЫ ПАЦИЕНТА: {symptoms}"
    )

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.1,  # low temperature for consistent clinical output
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model wraps output in ```json ... ```
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    parsed = json.loads(raw)
    return parsed.get("diagnoses", [])


# ---------------------------------------------------------------------------
# Fallback: if LLM fails, return retrieval-based results directly
# ---------------------------------------------------------------------------
def fallback_diagnoses(chunks: list[dict]) -> list[dict]:
    """Return top-5 results directly from retriever when LLM call fails."""
    diagnoses = []
    seen_protocols = set()
    rank = 1

    for chunk in chunks:
        pid = chunk["metadata"].get("protocol_id")
        if pid in seen_protocols:
            continue
        seen_protocols.add(pid)

        icd_codes = chunk["metadata"].get("icd_codes", [])
        if not icd_codes:
            continue

        diagnoses.append({
            "rank": rank,
            "icd10_code": icd_codes[0],
            "diagnosis": chunk["content"].split("\n")[0].strip("[]"),
            "explanation": chunk["content"][100:250].strip(),
        })
        rank += 1
        if rank > 5:
            break

    return diagnoses


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/diagnose", response_model=DiagnoseResponse)
async def handle_diagnose(request: DiagnoseRequest) -> DiagnoseResponse:
    symptoms = request.symptoms.strip()
    if not symptoms:
        return DiagnoseResponse(diagnoses=[])

    # Stage 2: Retrieve top-10 relevant protocol chunks
    retrieved = retriever.retrieve(symptoms, top_k=20, rerank_top_n=10)

    # Stage 3: Build context and call LLM
    context, available_icd_codes = build_context(retrieved)

    try:
        diagnoses_raw = call_llm(symptoms, context, available_icd_codes)
        diagnoses = [
            Diagnosis(
                rank=d.get("rank", i + 1),
                icd10_code=d.get("icd10_code", ""),
                diagnosis=d.get("diagnosis", ""),
                explanation=d.get("explanation", ""),
            )
            for i, d in enumerate(diagnoses_raw)
        ]
    except Exception as e:
        print(f"LLM call failed: {e} — falling back to retrieval-only results")
        fallback = fallback_diagnoses(retrieved)
        diagnoses = [Diagnosis(**d) for d in fallback]

    return DiagnoseResponse(diagnoses=diagnoses)


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Minimal web UI — text input → diagnosis results."""
    return HTMLResponse(content=HTML_UI)


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------
HTML_UI = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Медицинский диагностический ассистент</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 2rem;
    }
    .container { max-width: 760px; width: 100%; }
    h1 {
      font-size: 1.6rem;
      font-weight: 700;
      color: #38bdf8;
      margin-bottom: 0.4rem;
    }
    .subtitle {
      color: #94a3b8;
      font-size: 0.9rem;
      margin-bottom: 2rem;
    }
    textarea {
      width: 100%;
      min-height: 120px;
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 10px;
      padding: 1rem;
      color: #e2e8f0;
      font-size: 1rem;
      resize: vertical;
      outline: none;
      transition: border-color 0.2s;
    }
    textarea:focus { border-color: #38bdf8; }
    button {
      margin-top: 1rem;
      width: 100%;
      padding: 0.85rem;
      background: #0ea5e9;
      color: white;
      font-size: 1rem;
      font-weight: 600;
      border: none;
      border-radius: 10px;
      cursor: pointer;
      transition: background 0.2s;
    }
    button:hover { background: #0284c7; }
    button:disabled { background: #334155; cursor: not-allowed; }
    #results { margin-top: 2rem; }
    .card {
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 10px;
      padding: 1.2rem 1.4rem;
      margin-bottom: 1rem;
      animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; } }
    .card-header {
      display: flex;
      align-items: center;
      gap: 0.8rem;
      margin-bottom: 0.6rem;
    }
    .rank {
      background: #0ea5e9;
      color: white;
      font-weight: 700;
      border-radius: 6px;
      padding: 0.15rem 0.55rem;
      font-size: 0.85rem;
    }
    .icd-code {
      font-family: monospace;
      background: #0f172a;
      border: 1px solid #475569;
      border-radius: 5px;
      padding: 0.15rem 0.5rem;
      font-size: 0.9rem;
      color: #7dd3fc;
    }
    .diagnosis-name {
      font-weight: 600;
      font-size: 1rem;
      color: #f1f5f9;
    }
    .explanation {
      color: #94a3b8;
      font-size: 0.88rem;
      line-height: 1.5;
    }
    .loading {
      text-align: center;
      color: #38bdf8;
      padding: 2rem;
      font-size: 0.95rem;
    }
    .error {
      background: #450a0a;
      border: 1px solid #991b1b;
      color: #fca5a5;
      border-radius: 10px;
      padding: 1rem;
    }
  </style>
</head>
<body>
<div class="container">
  <h1>🏥 Диагностический ассистент</h1>
  <p class="subtitle">Казахстанские клинические протоколы · МКБ-10</p>
  <textarea id="symptoms" placeholder="Введите симптомы пациента на русском языке...&#10;&#10;Например: острая боль в правом подреберье, тошнота, срок беременности 32 недели"></textarea>
  <button id="submit-btn" onclick="diagnose()">Определить диагноз</button>
  <div id="results"></div>
</div>

<script>
async function diagnose() {
  const symptoms = document.getElementById('symptoms').value.trim();
  const resultsDiv = document.getElementById('results');
  const btn = document.getElementById('submit-btn');

  if (!symptoms) return;

  btn.disabled = true;
  btn.textContent = 'Анализирую...';
  resultsDiv.innerHTML = '<div class="loading">⏳ Поиск по протоколам и анализ симптомов...</div>';

  try {
    const response = await fetch('/diagnose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symptoms })
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();

    if (!data.diagnoses || data.diagnoses.length === 0) {
      resultsDiv.innerHTML = '<div class="error">Диагнозы не найдены. Попробуйте уточнить симптомы.</div>';
      return;
    }

    const sorted = [...data.diagnoses].sort((a, b) => a.rank - b.rank);
    resultsDiv.innerHTML = sorted.map(d => `
      <div class="card">
        <div class="card-header">
          <span class="rank">#${d.rank}</span>
          <span class="icd-code">${d.icd10_code}</span>
          <span class="diagnosis-name">${d.diagnosis}</span>
        </div>
        <p class="explanation">${d.explanation}</p>
      </div>
    `).join('');

  } catch (err) {
    resultsDiv.innerHTML = `<div class="error">Ошибка: ${err.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Определить диагноз';
  }
}

document.getElementById('symptoms').addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'Enter') diagnose();
});
</script>
</body>
</html>
"""