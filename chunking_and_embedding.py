import json
import re
import os
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from FlagEmbedding import BGEM3FlagModel
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------------
# DIAGNOSTIC WINDOW EXTRACTION
#
# Kazakhstan clinical protocols vary widely in structure, numbering, and
# section naming conventions. The one consistent landmark across all protocol
# types is:
#
#   START: first numbered/headed section containing "диагноз" or "диагностик"
#          (whichever appears first as a proper section header)
#   STOP:  first numbered section containing "лечени" or "цели лечения"
#          that comes AFTER the start
#
# Everything between these two landmarks is the diagnostic content window —
# symptoms, complaints, classification, differential diagnosis, physical exam,
# lab criteria. This is the only text we embed.
#
# If no landmarks are found, we fall back to the first 800 words of text.
# ---------------------------------------------------------------------------

# START: a line that begins a section (optionally numbered) containing
# диагноз or диагностик. Must appear as a section header, not inside prose.
# We match: optional newline, optional number/Roman prefix, then the keyword.
_START_PATTERN = re.compile(
    r'\n'
    r'(?:\d[\d\.]*\.?\s*|[IVXЛC]+\.?\s+)?'  # optional section number
    r'(?=[^\n]{0,60}\n)'                      # line must be short (header, not prose)
    r'[^\n]*[Дд]иагноз|'                      # contains диагноз
    r'\n'
    r'(?:\d[\d\.]*\.?\s*|[IVXЛC]+\.?\s+)?'
    r'[^\n]*[Дд]иагностик',                   # or диагностик
)

# Simpler, more reliable version: just find the earliest of the two keywords
# after skipping the title area (first 300 chars always contain the protocol
# title "Протокол диагностики и лечения <disease>" which would fire too early)
_DIAG_KW = re.compile(r'[Дд]иагност|[Дд]иагноз')

# STOP: numbered section header containing лечени or цели
# We require a newline + number to avoid stopping on prose mentions of лечение
_STOP_PATTERN = re.compile(
    r'\n\d+\.?\s*'
    r'(?:[Цц]ели\s+[Лл]ечени'      # "14. Цели лечения"
    r'|[Тт]актика\s+[Лл]ечени'     # "15. Тактика лечения"
    r'|[Лл]ечени[еяю][\s\n:]'       # "X. Лечение:" / "X. Лечения" / "X. Лечению"
    r')'
)

# Fallback stop: Roman numeral section with лечени (some older protocols)
_STOP_ROMAN = re.compile(
    r'\n[IVXЛC]+\.?\s+[^\n]*[Лл]ечени'
)

# Minimum diagnostic window size — if we extract something shorter than this
# (in characters), something probably went wrong and we use fallback
_MIN_WINDOW_CHARS = 50   # even a short section is better than 800-word fallback

# If the window is longer than this (tokens), also produce a focused
# "head chunk" containing just the first HEAD_TOKENS tokens.
# The head almost always contains the жалобы/complaints section.
_HEAD_TOKENS = 300


def extract_diagnostic_window(text: str) -> tuple[str, str]:
    """
    Extract the diagnostic window from a protocol's plain text.

    Returns (window_text, strategy) where strategy is one of:
      'landmark'  — found both start and stop landmarks
      'start_only' — found start but no stop (used rest of text)
      'fallback'   — no landmarks found, used first 800 words

    The returned text is stripped but otherwise unmodified.
    """
    # Skip the title area (first ~300 chars) to avoid matching the protocol
    # title "Протокол диагностики и лечения <disease>"
    # Skip the approval header — all protocols start with an administrative
    # paragraph (commission approval, date, protocol number) followed by a
    # blank line. The actual numbered content begins after that first blank line.
    # Using \n\n as the boundary is robust across both old and new protocol styles.
    first_blank = text.find('\n\n')
    title_skip = first_blank + 2 if first_blank >= 0 else 150

    # Find start: search for the keyword after the title area, but require it
    # to appear at the START of a line (preceded only by optional whitespace and
    # an optional section number). This avoids mid-sentence matches like
    # "по диагностике и лечению" in citation text.
    #
    # Two-step approach because pure regex character encoding issues make
    # embedded Cyrillic in raw string patterns unreliable:
    # Step 1: find any keyword occurrence
    # Step 2: check that only a section number (or nothing) precedes it on the line
    _section_prefix = re.compile(r'^[ \t]*(?:\d[\d.]*[.]?\s+|[IVXЛC]+[.]?\s+)?$')
    start_pos = None
    for _m in _DIAG_KW.finditer(text, title_skip):
        _lb = text.rfind('\n', 0, _m.start())
        _line_start = _lb + 1 if _lb >= 0 else 0
        _before = text[_line_start:_m.start()]
        if _section_prefix.match(_before):
            start_pos = _line_start
            break
    if start_pos is None:
        # No well-formed section header found — use any occurrence as fallback
        any_match = _DIAG_KW.search(text, title_skip)
        if not any_match:
            words = text.split()
            return ' '.join(words[:800]), 'fallback'
        lb = text.rfind('\n', 0, any_match.start())
        start_pos = lb + 1 if lb >= 0 else any_match.start()

    # Find stop: first matching numbered section after start_pos
    stop_match = _STOP_PATTERN.search(text, start_pos)
    if not stop_match:
        # Try Roman numeral fallback stop
        stop_match = _STOP_ROMAN.search(text, start_pos)

    if stop_match:
        window = text[start_pos:stop_match.start()].strip()
        strategy = 'landmark'
    else:
        # No stop found — use everything from start to end
        window = text[start_pos:].strip()
        strategy = 'start_only'

    if len(window) < _MIN_WINDOW_CHARS:
        # Extraction produced almost nothing — fall back
        words = text.split()
        return ' '.join(words[:800]), 'fallback'

    return window, strategy


def build_chunks_from_window(protocol_title: str, icd_codes: list[str],
                              window: str, protocol_id: str) -> list[dict]:
    """
    Given the extracted diagnostic window, produce 1 or 2 chunks:

    1. Full window chunk — the entire diagnostic window.
       Used for recall: ensures all diagnostic content is reachable.

    2. Head chunk (only if window > _HEAD_TOKENS tokens) — the first
       _HEAD_TOKENS tokens of the window, which typically contains
       the жалобы/complaints section.
       Used for precision: clean symptom signal without table noise.

    Both chunks are prefixed with protocol title and ICD codes.
    """
    prefix = protocol_title
    if icd_codes:
        prefix += f"\nМКБ-10: {', '.join(icd_codes)}"

    meta = {'protocol_id': protocol_id, 'icd_codes': icd_codes}
    chunks = []

    # Chunk 1: full window
    full_content = prefix + '\n\n' + window
    chunks.append({
        'content': full_content,
        'chunk_type': 'diagnostic_window',
        'metadata': {**meta, 'section': 'diagnostic_window'},
    })

    # Chunk 2: head-only (first HEAD_TOKENS tokens) if window is long
    window_tokens = window.split()
    if len(window_tokens) > _HEAD_TOKENS:
        head_text = ' '.join(window_tokens[:_HEAD_TOKENS])
        head_content = prefix + '\n\n' + head_text
        chunks.append({
            'content': head_content,
            'chunk_type': 'diagnostic_head',
            'metadata': {**meta, 'section': 'diagnostic_head'},
        })

    return chunks


def get_structural_chunks_from_dict(data: dict) -> list[dict]:
    """
    Main entry point. Given one JSONL protocol record, extract the diagnostic
    window and return 1–2 embeddable chunks.
    """
    protocol_id = data.get('protocol_id', 'unknown')
    icd_codes = data.get('icd_codes', [])
    text = data.get('text', '')
    protocol_title = data.get('source_file', 'Unknown').replace('.pdf', '')

    window, strategy = extract_diagnostic_window(text)
    chunks = build_chunks_from_window(protocol_title, icd_codes, window, protocol_id)

    # Attach extraction strategy to metadata for debugging
    for c in chunks:
        c['metadata']['extraction_strategy'] = strategy

    return chunks


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def debug_protocol(data: dict):
    """Print the extracted window and chunks for a single protocol record."""
    protocol_title = data.get('source_file', 'Unknown')
    text = data.get('text', '')

    window, strategy = extract_diagnostic_window(text)
    window_tokens = len(window.split())

    print(f"\n{'='*70}")
    print(f"Protocol  : {protocol_title}")
    print(f"ICD codes : {data.get('icd_codes', [])}")
    print(f"Strategy  : {strategy}")
    print(f"Window    : {window_tokens} tokens / {len(window)} chars")
    print(f"{'='*70}")
    print("\n--- WINDOW START ---")
    print(window[:400])
    print("\n... [middle omitted] ...\n")
    print("--- WINDOW END ---")
    print(window[-300:])

    chunks = get_structural_chunks_from_dict(data)
    print(f"\nChunks produced: {len(chunks)}")
    for c in chunks:
        print(f"  {c['chunk_type']:25s} | {len(c['content'].split()):4d} tokens")


def debug_corpus_stats(jsonl_path: str, max_protocols: int = 50):
    """
    Quick scan of the corpus to check window extraction quality.
    Prints per-protocol stats and a summary.
    """
    strategies = {'landmark': 0, 'start_only': 0, 'fallback': 0}
    token_counts = []
    fallback_protocols = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_protocols:
                break
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            window, strategy = extract_diagnostic_window(data.get('text', ''))
            strategies[strategy] = strategies.get(strategy, 0) + 1
            token_counts.append(len(window.split()))
            if strategy == 'fallback':
                fallback_protocols.append(data.get('source_file', '?'))

    print(f"\nCorpus sample stats ({max_protocols} protocols):")
    print(f"  landmark   : {strategies.get('landmark', 0)}")
    print(f"  start_only : {strategies.get('start_only', 0)}")
    print(f"  fallback   : {strategies.get('fallback', 0)}")
    if token_counts:
        print(f"  window tokens — min: {min(token_counts)}, "
              f"max: {max(token_counts)}, "
              f"avg: {sum(token_counts)//len(token_counts)}")
    if fallback_protocols:
        print(f"  fallback protocols: {fallback_protocols[:10]}")


# ---------------------------------------------------------------------------
# GPU encoding
# ---------------------------------------------------------------------------

def encode_on_gpu(gpu_id: int, chunks: list[dict]) -> dict:
    if not chunks:
        return {'dense_vecs': np.array([]), 'lexical_weights': []}
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    texts = [c['content'] for c in chunks]
    out = model.encode(texts, return_dense=True, return_sparse=True)
    return {'dense_vecs': out['dense_vecs'], 'lexical_weights': out['lexical_weights']}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    NUM_GPUS = 5  # adjust to your hardware

    jsonl_file_path = 'corpus/protocols_corpus.jsonl'

    # Optional: run a quick corpus quality check before full encoding
    print("=== Corpus diagnostic window extraction stats ===")
    debug_corpus_stats(jsonl_file_path, max_protocols=9999)

    # 1. Parse all protocols into chunks
    all_chunks = []
    protocol_count = 0
    type_counts: dict[str, int] = {}
    strategy_counts: dict[str, int] = {}

    print("\nParsing protocols...")
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            protocol_count += 1
            new_chunks = get_structural_chunks_from_dict(data)
            all_chunks.extend(new_chunks)
            for c in new_chunks:
                ct = c['chunk_type']
                type_counts[ct] = type_counts.get(ct, 0) + 1
                st = c['metadata']['extraction_strategy']
                strategy_counts[st] = strategy_counts.get(st, 0) + 1

    print(f"\nProtocols    : {protocol_count}")
    print(f"Total chunks : {len(all_chunks)}  "
          f"(avg {len(all_chunks)/protocol_count:.1f} per protocol)")
    print("Chunk types:")
    for ct, cnt in sorted(type_counts.items()):
        print(f"  {ct:25s}: {cnt:5d}")
    print("Extraction strategies:")
    for st, cnt in sorted(strategy_counts.items()):
        print(f"  {st:25s}: {cnt:5d}")
    avg_len = sum(len(c['content'].split()) for c in all_chunks) / max(len(all_chunks), 1)
    print(f"Avg chunk length: {avg_len:.0f} tokens")

    # 2. BM25
    print("\nBuilding BM25 index...")
    bm25 = BM25Okapi([c['content'].split() for c in all_chunks])
    with open('bm25_index.pkl', 'wb') as f:
        pickle.dump(bm25, f)
    print("→ bm25_index.pkl")

    # 3. Split for GPUs
    k, m = divmod(len(all_chunks), NUM_GPUS)
    splits = [
        all_chunks[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
        for i in range(NUM_GPUS)
    ]
    print(f"\nGPU split: {[len(s) for s in splits]}")

    # 4. Encode
    with ProcessPoolExecutor(max_workers=NUM_GPUS) as ex:
        futures = [ex.submit(encode_on_gpu, i, splits[i]) for i in range(NUM_GPUS)]
        results = [f.result() for f in futures]

    # 5. Merge and save
    dense = np.concatenate(
        [r['dense_vecs'] for r in results if len(r['dense_vecs'])], axis=0
    )
    sparse = []
    for r in results:
        sparse.extend(r['lexical_weights'])

    with open('processed_corpus.json', 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    np.savez('embeddings.npz', dense=dense)
    with open('sparse_weights.json', 'w', encoding='utf-8') as f:
        json.dump(
            [{t: float(w) for t, w in doc.items()} for doc in sparse],
            f, ensure_ascii=False
        )

    print(f"\nSaved {len(all_chunks)} chunks:")
    print("  → processed_corpus.json")
    print("  → embeddings.npz")
    print("  → sparse_weights.json")
    print("  → bm25_index.pkl")