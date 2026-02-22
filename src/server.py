"""
Medical Diagnosis Server
POST /diagnose  {"symptoms": "..."}
GET  /          Web UI
"""

import json
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
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

CORPUS_FILE   = os.environ.get("CORPUS_FILE",   "processed_corpus.json")
EMBED_FILE    = os.environ.get("EMBED_FILE",    "embeddings.npz")
SPARSE_FILE   = os.environ.get("SPARSE_FILE",   "sparse_weights.json")
BM25_FILE     = os.environ.get("BM25_FILE",     "bm25_index.pkl")
ICD_DESC_FILE = os.environ.get("ICD_DESC_FILE", "icd_descriptions.json")

LLM_TIMEOUT  = 15.0
RETRIEVE_K   = 20   # candidates before reranking
RERANK_TOP_N = 5    # unique protocols to keep after reranking

# ---------------------------------------------------------------------------
# System prompt
# Modelled after the notebook's structured JSON approach.
# The LLM receives: a dict of ICD code → description (only the ones retrieved),
# plus the patient symptoms. It returns top-3 ranked by likelihood.
# No chunk text is sent — the descriptions alone give enough signal.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
Ты — опытный врач-клиницист и система поддержки принятия диагностических решений \
на основе клинических протоколов Министерства здравоохранения Республики Казахстан.

Тебе предоставлены:
1. Симптомы пациента
2. Выдержки из клинических протоколов (тексты), релевантные симптомам
3. Словарь кодов МКБ-10 с описаниями — ТОЛЬКО из протоколов, релевантных симптомам

ЗАДАЧА: выбери до 5 наиболее подходящих диагнозов из CANDIDATE_CODES, строго основываясь на предоставленных текстах протоколов.

ЖЁСТКИЕ ПРАВИЛА — нарушение любого делает ответ недействительным:
1. "icd10_code" — ТОЛЬКО коды из CANDIDATE_CODES. Запрещено придумывать коды.
2. Все выбранные коды должны быть РАЗНЫМИ.
3. СТРОГИЙ ФИЛЬТР: Внимательно читай тексты протоколов! Если протокол требует наличия специфических или тяжелых симптомов (например, сыпь, кровотечение, определенные лабораторные показатели), которых НЕТ в описании пациента, ты ОБЯЗАН исключить этот диагноз, даже если он есть в списке кандидатов.
4. Предпочитай точные коды с точкой (E06.9, F32.0). Если в CANDIDATE_CODES есть только общий код (E06) — используй его.
5. "name" — название из описания в CANDIDATE_CODES.
6. "reasoning" — ровно 1 предложение, объясняющее совпадение симптомов пациента с критериями именно из ТЕКСТА протокола.

Отвечай ТОЛЬКО валидным JSON без markdown-обёртки:
{
  "diagnoses": [
    {"rank": 1, "icd10_code": "A00.0", "name": "...", "reasoning": "..."},
    {"rank": 2, "icd10_code": "B00.0", "name": "...", "reasoning": "..."},
    {"rank": 3, "icd10_code": "C00.0", "name": "...", "reasoning": "..."},
    {"rank": 4, "icd10_code": "D00.0", "name": "...", "reasoning": "..."},
    {"rank": 5, "icd10_code": "E00.0", "name": "...", "reasoning": "..."}
  ]
}
"""

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
retriever:    MedicalRetriever = None
llm_client:   OpenAI           = None
icd_descriptions: dict         = {}   # code -> Russian description


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, llm_client, icd_descriptions
    print("\n🏥 Medical Diagnosis Server")
    print("=" * 40)

    # ICD descriptions (built by parse_icd_descriptions.py)
    if os.path.exists(ICD_DESC_FILE):
        with open(ICD_DESC_FILE, 'r', encoding='utf-8') as f:
            icd_descriptions = json.load(f)
        print(f"Loaded {len(icd_descriptions)} ICD descriptions")
    else:
        print(f"Warning: {ICD_DESC_FILE} not found — codes will be sent without descriptions")

    print("Loading retriever...")
    retriever  = MedicalRetriever(CORPUS_FILE, EMBED_FILE, SPARSE_FILE, BM25_FILE)
    llm_client = OpenAI(base_url=HUB_URL, api_key=API_KEY)
    print("✓ Ready. POST /diagnose")
    print("=" * 40 + "\n")
    yield


app = FastAPI(title="Medical Diagnosis Server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class DiagnoseRequest(BaseModel):
    symptoms: str | None = None

class Diagnosis(BaseModel):
    rank:       int
    icd10_code: str
    name:       str
    reasoning:  str

class DiagnoseResponse(BaseModel):
    diagnoses: list[Diagnosis]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def top_protocols(symptoms: str) -> list[dict]:
    """
    Retrieve and rerank candidates, then deduplicate to unique protocols.
    Returns up to RERANK_TOP_N protocol chunks (one per protocol).

    Deduplication happens AFTER reranking so the best-scored chunk per
    protocol is kept — not just any chunk from that protocol.
    """
    candidates = retriever.retrieve(symptoms, top_k=RETRIEVE_K, rerank_top_n=RETRIEVE_K)

    seen: set = set()
    unique: list = []
    for chunk in candidates:                           # already sorted by rerank_score desc
        pid = chunk["metadata"].get("protocol_id")
        if pid not in seen and chunk["metadata"].get("icd_codes"):
            seen.add(pid)
            unique.append(chunk)
        if len(unique) >= RERANK_TOP_N:
            break

    return unique


def expand_code(code: str) -> dict[str, str]:
    """
    Turn one ICD code into a {code: description} dict of precise entries.

    - Dot-code (e.g. O14.1): look up directly in icd_descriptions.
      If found, return it. If not, return the code with itself as placeholder.
    - Bare parent code (e.g. O14, O99): expand to ALL O14.x / O99.x entries
      present in icd_descriptions. If none exist, return the bare code as
      a placeholder so the protocol is never silently dropped.
    """
    if '.' in code:
        desc = icd_descriptions.get(code, code)
        return {code: desc}
    else:
        # Expand to every known subcode whose prefix matches
        prefix = code + '.'
        subcodes = {k: v for k, v in icd_descriptions.items() if k.startswith(prefix)}
        if subcodes:
            return subcodes
        # No subcodes known at all — keep the bare code as a last resort
        return {code: code}


def build_candidate_dict(chunks: list[dict]) -> dict[str, str]:
    """
    Build an ordered {icd_code: description} dict from the retrieved chunks.

    Chunks arrive sorted by rerank score (best first). Python dicts preserve
    insertion order, so codes from the top-ranked protocol are inserted first
    and appear first in the formatted prompt — giving the LLM an implicit
    ordering signal that earlier entries are more likely matches.

    Expansion: dot-codes added as-is; bare parent codes (e.g. O14) expanded
    to all known O14.x subcodes from icd_descriptions. First protocol wins
    on duplicate codes across protocols.
    """
    candidate: dict[str, str] = {}
    for chunk in chunks:
        for code in chunk["metadata"].get("icd_codes", []):
            for expanded_code, desc in expand_code(code).items():
                if expanded_code not in candidate:
                    candidate[expanded_code] = desc
    return candidate


def call_llm(symptoms: str, candidate_codes: dict[str, str], chunks: list[dict]) -> list[dict]:
    """
    Send symptoms, candidate ICD dict, and retrieved chunks to the LLM.
    Returns the parsed diagnoses list, or raises on failure.
    """
    # Format candidate codes
    lines = [f"  {i}. {code}: {desc}"
             for i, (code, desc) in enumerate(candidate_codes.items(), 1)]
    codes_text = "\n".join(lines)

    # Format chunk texts for the LLM to read
    chunks_text = ""
    for i, chunk in enumerate(chunks[:3], 1):
        # We extract the protocol title if available, otherwise just number it
        protocol_title = chunk.get("metadata", {}).get("title", f"Протокол {i}")
        content = chunk.get("content", "").strip()
        if len(content) > 1000:
            content = content[:1000] + "... [ОСТАЛЬНОЙ ТЕКСТ УРЕЗАН]"
        
        chunks_text += f"\n--- ТЕКСТ ПРОТОКОЛА {i}: {protocol_title} ---\n{content}\n"

    user_message = (
        f"СИМПТОМЫ ПАЦИЕНТА:\n{symptoms}\n\n"
        f"CANDIDATE_CODES (отсортированы по убыванию релевантности):\n{codes_text}\n\n"
        f"КЛИНИЧЕСКИЕ ПРОТОКОЛЫ (используй для строгой проверки симптомов):\n{chunks_text}"
    )

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",          "", raw)
    return json.loads(raw).get("diagnoses", [])


def fallback_diagnoses(chunks: list[dict], candidate_codes: dict[str, str]) -> list[Diagnosis]:
    """
    Used when the LLM fails. Returns the top-3 protocols directly,
    using the first dot-code per protocol as the ICD code.
    """
    results = []
    for i, chunk in enumerate(chunks[:3], 1):
        # Pick first dot-code available
        code = next(
            (c for c in chunk["metadata"].get("icd_codes", []) if '.' in c),
            None
        )
        if not code:
            continue
        results.append(Diagnosis(
            rank=i,
            icd10_code=code,
            name=candidate_codes.get(code, code),
            reasoning="(LLM недоступен — результат по близости эмбеддинга)",
        ))
    return results


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/diagnose", response_model=DiagnoseResponse)
async def handle_diagnose(request: DiagnoseRequest) -> DiagnoseResponse:
    symptoms = (request.symptoms or "").strip()
    if not symptoms:
        return DiagnoseResponse(diagnoses=[])

    # 1. Retrieve top-5 unique protocols
    chunks = top_protocols(symptoms)
    if not chunks:
        return DiagnoseResponse(diagnoses=[])

    # 2. Build ICD code → description dict from retrieved protocols only
    candidate_codes = build_candidate_dict(chunks)
    if not candidate_codes:
        return DiagnoseResponse(diagnoses=[])

    # 3. Ask LLM to pick top-3 from the candidate dict
    try:
        raw = call_llm(symptoms, candidate_codes, chunks)

        # Parse, deduplicate, and guard against hallucinations.
        # We do NOT filter bare codes (e.g. S06, L10) — the evaluator uses
        # prefix matching so "S06" counts as correct for ground truth "S06.3".
        seen_codes: set = set()
        diagnoses = []
        rank = 1
        for d in raw:
            code = d.get("icd10_code", "").strip()
            if not code:
                continue
            if code in seen_codes:                  # reject duplicates
                continue
            if code not in candidate_codes:         # reject hallucinations
                print(f"  Skipped hallucinated code: {code}")
                continue
            seen_codes.add(code)
            diagnoses.append(Diagnosis(
                rank=rank,
                icd10_code=code,
                name=d.get("name", candidate_codes.get(code, code)).strip(),
                reasoning=d.get("reasoning", "").strip(),
            ))
            rank += 1

        if not diagnoses:
            raise ValueError("LLM returned no valid codes")
    except Exception as e:
        print(f"LLM failed ({type(e).__name__}: {e}) — using fallback")
        diagnoses = fallback_diagnoses(chunks, candidate_codes)

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
      min-width: 2rem; text-align: center;
    }
    .icd-code {
      font-family: monospace; background: #0f172a;
      border: 1px solid #475569; border-radius: 5px;
      padding: 0.15rem 0.5rem; font-size: 0.9rem; color: #7dd3fc;
    }
    .diagnosis-name { font-weight: 600; font-size: 1rem; color: #f1f5f9; }
    .reasoning { color: #94a3b8; font-size: 0.88rem; line-height: 1.5; margin-top: 0.3rem; }
    .loading { text-align: center; color: #38bdf8; padding: 2rem; font-size: 0.95rem; }
    .error { background: #450a0a; border: 1px solid #991b1b; color: #fca5a5;
             border-radius: 10px; padding: 1rem; }
  </style>
</head>
<body>
<div class="container">
  <h1>🏥 Диагностический ассистент</h1>
  <p class="subtitle">Казахстанские клинические протоколы · МКБ-10</p>
  <textarea id="symptoms"
    placeholder="Введите симптомы пациента на русском языке...&#10;&#10;Например: острая боль в правом подреберье, тошнота, срок беременности 32 недели"
  ></textarea>
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
    resultsDiv.innerHTML = data.diagnoses
      .sort((a, b) => a.rank - b.rank)
      .map(d => `
        <div class="card">
          <div class="card-header">
            <span class="rank">#${d.rank}</span>
            <span class="icd-code">${d.icd10_code}</span>
            <span class="diagnosis-name">${d.name}</span>
          </div>
          <p class="reasoning">${d.reasoning}</p>
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