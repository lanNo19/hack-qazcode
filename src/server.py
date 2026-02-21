"""
Medical Diagnosis Server
POST /diagnose  {"symptoms": "..."}
GET  /          Web UI
"""

import json
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI

import sys
sys.path.insert(0, os.path.dirname(__file__))
from retrieval_and_reranking import MedicalRetriever

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HUB_URL   = os.environ.get("HUB_URL", "YOUR_HUB_URL")
API_KEY   = os.environ.get("API_KEY", "YOUR_API_KEY")
LLM_MODEL = "oss-120b"

CORPUS_FILE = os.environ.get("CORPUS_FILE", "processed_corpus.json")
EMBED_FILE  = os.environ.get("EMBED_FILE",  "embeddings.npz")
SPARSE_FILE = os.environ.get("SPARSE_FILE", "sparse_weights.json")
BM25_FILE   = os.environ.get("BM25_FILE",   "bm25_index.pkl")

# FIX 1: Strict LLM timeout — prevents 40s hangs that tank latency and count as errors
LLM_TIMEOUT = 15.0

# How many chunks to retrieve and pass to the LLM
# FIX 2: Reduced from 10 to 5 — less context = LLM follows instructions better
RERANK_TOP_N = 5

# ---------------------------------------------------------------------------
# System prompt
# FIX 3: Much stricter prompt — explicitly forbids using chunk text as diagnosis name,
#         requires chain-of-thought before ranking, constrains to AVAILABLE_ICD_CODES
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
Ты — система клинической поддержки принятия решений на основе клинических протоколов МЗ РК.

Тебе будут даны симптомы пациента и несколько фрагментов протоколов с кодами МКБ-10.

СТРОГИЕ ПРАВИЛА:
1. Поле "icd10_code" — ТОЛЬКО код из списка AVAILABLE_ICD_CODES. Никаких других кодов.
2. Поле "diagnosis" — краткое медицинское название болезни (2-6 слов). НЕ копируй текст протокола.
3. Поле "explanation" — 1-2 предложения: какие симптомы пациента совпадают с критериями протокола.
4. Верни ровно 5 диагнозов, ранжированных по убыванию вероятности (rank 1 = наиболее вероятный).

ПРОЦЕСС (думай шаг за шагом):
- Для каждого протокола: совпадают ли симптомы пациента с диагностическими критериями?
- Выбери наиболее подходящий код МКБ-10 из AVAILABLE_ICD_CODES для каждого протокола.
- Поставь на rank 1 протокол с наибольшим совпадением симптомов.

Отвечай ТОЛЬКО валидным JSON. Никакого markdown, никаких пояснений вне JSON:
{
  "diagnoses": [
    {"rank": 1, "icd10_code": "X00.0", "diagnosis": "Название болезни", "explanation": "Обоснование."},
    {"rank": 2, "icd10_code": "Y00.0", "diagnosis": "Название болезни", "explanation": "Обоснование."},
    {"rank": 3, "icd10_code": "Z00.0", "diagnosis": "Название болезни", "explanation": "Обоснование."},
    {"rank": 4, "icd10_code": "A00.0", "diagnosis": "Название болезни", "explanation": "Обоснование."},
    {"rank": 5, "icd10_code": "B00.0", "diagnosis": "Название болезни", "explanation": "Обоснование."}
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
    print("Loading retriever...")
    retriever = MedicalRetriever(CORPUS_FILE, EMBED_FILE, SPARSE_FILE, BM25_FILE)
    llm_client = OpenAI(base_url=HUB_URL, api_key=API_KEY, timeout=LLM_TIMEOUT)
    print("✓ Ready. POST /diagnose")
    print("=" * 40 + "\n")
    yield


app = FastAPI(title="Medical Diagnosis Server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Models
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
# Helpers
# ---------------------------------------------------------------------------
def build_context(chunks: list[dict]) -> tuple[str, list[str]]:
    """Format chunks for LLM. Returns (context_text, flat_icd_code_list)."""
    parts = []
    all_icd = []

    for i, chunk in enumerate(chunks, 1):
        icd_codes = chunk["metadata"].get("icd_codes", [])
        all_icd.extend(icd_codes)
        # FIX 4: Only send the first 400 chars of each chunk — enough for diagnostic criteria,
        # cuts context length ~60% which makes the LLM much more reliable
        content_preview = chunk["content"][:400].strip()
        parts.append(
            f"[Протокол {i} | МКБ-10: {', '.join(icd_codes) or 'н/д'}]\n{content_preview}"
        )

    seen = set()
    unique_icd = [c for c in all_icd if not (c in seen or seen.add(c))]
    return "\n\n".join(parts), unique_icd


def call_llm(symptoms: str, context: str, available_icd: list[str]) -> list[dict]:
    user_message = (
        f"AVAILABLE_ICD_CODES: {', '.join(available_icd)}\n\n"
        f"ПРОТОКОЛЫ:\n{context}\n\n"
        f"СИМПТОМЫ ПАЦИЕНТА: {symptoms}"
    )

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.0,  # fully deterministic for clinical output
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw).get("diagnoses", [])


def fallback_diagnoses(chunks: list[dict]) -> list[Diagnosis]:
    """
    FIX 5: Better fallback — when LLM fails, return the reranked retrieval results
    with clean diagnosis names extracted from the context string (not raw chunk text).
    """
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

        # Extract clean protocol name from context string e.g. "[HELLP-СИНДРОМ > ...]"
        context_match = re.match(r'\[([^\]>]+)', chunk["content"])
        protocol_name = context_match.group(1).strip() if context_match else "Неизвестно"

        # Find first sentence of actual content as explanation
        lines = chunk["content"].split("\n")
        explanation = next((l.strip() for l in lines[1:] if len(l.strip()) > 20), "")[:200]

        diagnoses.append(Diagnosis(
            rank=rank,
            icd10_code=icd_codes[0],
            diagnosis=protocol_name,
            explanation=explanation,
        ))
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

    retrieved = retriever.retrieve(symptoms, top_k=20, rerank_top_n=RERANK_TOP_N)
    context, available_icd = build_context(retrieved)

    try:
        raw_diagnoses = call_llm(symptoms, context, available_icd)
        diagnoses = [
            Diagnosis(
                rank=d.get("rank", i + 1),
                icd10_code=d.get("icd10_code", "").strip(),
                diagnosis=d.get("diagnosis", "").strip(),
                explanation=d.get("explanation", "").strip(),
            )
            for i, d in enumerate(raw_diagnoses)
        ]
        if not diagnoses:
            raise ValueError("LLM returned empty diagnoses list")
    except Exception as e:
        print(f"LLM failed ({type(e).__name__}: {e}) — using fallback")
        diagnoses = fallback_diagnoses(retrieved)

    return DiagnoseResponse(diagnoses=diagnoses)


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
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
      background: #0f172a; color: #e2e8f0;
      min-height: 100vh; display: flex;
      align-items: center; justify-content: center; padding: 2rem;
    }
    .container { max-width: 760px; width: 100%; }
    h1 { font-size: 1.6rem; font-weight: 700; color: #38bdf8; margin-bottom: 0.4rem; }
    .subtitle { color: #94a3b8; font-size: 0.9rem; margin-bottom: 2rem; }
    textarea {
      width: 100%; min-height: 120px; background: #1e293b;
      border: 1px solid #334155; border-radius: 10px;
      padding: 1rem; color: #e2e8f0; font-size: 1rem;
      resize: vertical; outline: none; transition: border-color 0.2s;
    }
    textarea:focus { border-color: #38bdf8; }
    button {
      margin-top: 1rem; width: 100%; padding: 0.85rem;
      background: #0ea5e9; color: white; font-size: 1rem;
      font-weight: 600; border: none; border-radius: 10px;
      cursor: pointer; transition: background 0.2s;
    }
    button:hover { background: #0284c7; }
    button:disabled { background: #334155; cursor: not-allowed; }
    #results { margin-top: 2rem; }
    .card {
      background: #1e293b; border: 1px solid #334155;
      border-radius: 10px; padding: 1.2rem 1.4rem;
      margin-bottom: 1rem; animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; } }
    .card-header { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.6rem; }
    .rank {
      background: #0ea5e9; color: white; font-weight: 700;
      border-radius: 6px; padding: 0.15rem 0.55rem; font-size: 0.85rem;
    }
    .icd-code {
      font-family: monospace; background: #0f172a;
      border: 1px solid #475569; border-radius: 5px;
      padding: 0.15rem 0.5rem; font-size: 0.9rem; color: #7dd3fc;
    }
    .diagnosis-name { font-weight: 600; font-size: 1rem; color: #f1f5f9; }
    .explanation { color: #94a3b8; font-size: 0.88rem; line-height: 1.5; }
    .loading { text-align: center; color: #38bdf8; padding: 2rem; font-size: 0.95rem; }
    .error { background: #450a0a; border: 1px solid #991b1b; color: #fca5a5; border-radius: 10px; padding: 1rem; }
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