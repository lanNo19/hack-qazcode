import json
import re
import os
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from FlagEmbedding import BGEM3FlagModel
from rank_bm25 import BM25Okapi
from razdel import sentenize


# ---------------------------------------------------------------------------
# DIAGNOSTIC WINDOW EXTRACTION
# ---------------------------------------------------------------------------
_START_PATTERN = re.compile(
    r'\n'
    r'(?:\d[\d\.]*\.?\s*|[IVXЛC]+\.?\s+)?'
    r'(?=[^\n]{0,60}\n)'
    r'[^\n]*[Дд]иагноз|'
    r'\n'
    r'(?:\d[\d\.]*\.?\s*|[IVXЛC]+\.?\s+)?'
    r'[^\n]*[Дд]иагностик',
)

_DIAG_KW = re.compile(r'[Дд]иагност|[Дд]иагноз')

_STOP_PATTERN = re.compile(
    r'\n\d+\.?\s*'
    r'(?:[Цц]ели\s+[Лл]ечени'
    r'|[Тт]актика\s+[Лл]ечени'
    r'|[Лл]ечени[еяю][\s\n:]'
    r')'
)

_STOP_ROMAN = re.compile(
    r'\n[IVXЛC]+\.?\s+[^\n]*[Лл]ечени'
)

_MIN_WINDOW_CHARS = 50


def extract_diagnostic_window(text: str) -> tuple[str, str]:
    """Extract the diagnostic window from a protocol's plain text."""
    first_blank = text.find('\n\n')
    title_skip = first_blank + 2 if first_blank >= 0 else 150

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
        any_match = _DIAG_KW.search(text, title_skip)
        if not any_match:
            words = text.split()
            return ' '.join(words[:800]), 'fallback'
        lb = text.rfind('\n', 0, any_match.start())
        start_pos = lb + 1 if lb >= 0 else any_match.start()

    stop_match = _STOP_PATTERN.search(text, start_pos)
    if not stop_match:
        stop_match = _STOP_ROMAN.search(text, start_pos)

    if stop_match:
        window = text[start_pos:stop_match.start()].strip()
        strategy = 'landmark'
    else:
        window = text[start_pos:].strip()
        strategy = 'start_only'

    if len(window) < _MIN_WINDOW_CHARS:
        words = text.split()
        return ' '.join(words[:800]), 'fallback'

    return window, strategy


# ---------------------------------------------------------------------------
# SEMANTIC CHUNKING LOGIC
# ---------------------------------------------------------------------------

def chunk_text_semantically(text: str, max_words: int = 200, overlap_words: int = 50) -> list[str]:
    """Splits Russian text into semantic chunks based on sentence boundaries using razdel."""
    sentences = [s.text for s in sentenize(text)]
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)
        
        # If adding this sentence exceeds our limit, save the chunk and slide the window
        if current_word_count + word_count > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # Keep the last few sentences to create our overlap window
            while current_word_count > overlap_words and len(current_chunk) > 1:
                dropped_sentence = current_chunk.pop(0)
                current_word_count -= len(dropped_sentence.split())
                
        current_chunk.append(sentence)
        current_word_count += word_count
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks


def build_semantic_chunks_from_window(protocol_title: str, icd_codes: list[str],
                                      window: str, protocol_id: str) -> list[dict]:
    """
    Given the extracted diagnostic window, produce semantic chunks.
    Each chunk is prepended with the protocol title and ICD codes.
    """
    prefix = protocol_title
    if icd_codes:
        prefix += f"\nМКБ-10: {', '.join(icd_codes)}"

    meta = {'protocol_id': protocol_id, 'icd_codes': icd_codes}
    chunks = []

    text_chunks = chunk_text_semantically(window, max_words=200, overlap_words=50)
    
    for i, text_chunk in enumerate(text_chunks):
        content = prefix + '\n\n' + text_chunk
        chunks.append({
            'content': content,
            'chunk_type': 'semantic_chunk',
            'metadata': {**meta, 'chunk_index': i}
        })

    return chunks


def get_semantic_chunks_from_dict(data: dict) -> list[dict]:
    """
    Main entry point. Given one JSONL protocol record, extract the diagnostic
    window and return its semantic chunks.
    """
    protocol_id = data.get('protocol_id', 'unknown')
    icd_codes = data.get('icd_codes', [])
    text = data.get('text', '')
    protocol_title = data.get('source_file', 'Unknown').replace('.pdf', '')

    window, strategy = extract_diagnostic_window(text)
    chunks = build_semantic_chunks_from_window(protocol_title, icd_codes, window, protocol_id)

    # Attach extraction strategy to metadata for debugging
    for c in chunks:
        c['metadata']['extraction_strategy'] = strategy

    return chunks


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
    NUM_GPUS = 1  # ADJUST TO YOUR HARDWARE (set back to 5 if needed)

    jsonl_file_path = 'corpus/protocols_corpus.jsonl'

    # 1. Parse all protocols into chunks
    all_chunks = []
    protocol_count = 0
    strategy_counts: dict[str, int] = {}

    print("\nParsing protocols into semantic chunks...")
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            protocol_count += 1
            new_chunks = get_semantic_chunks_from_dict(data)
            all_chunks.extend(new_chunks)
            for c in new_chunks:
                st = c['metadata']['extraction_strategy']
                strategy_counts[st] = strategy_counts.get(st, 0) + 1

    print(f"\nProtocols    : {protocol_count}")
    print(f"Total chunks : {len(all_chunks)}  "
          f"(avg {len(all_chunks)/protocol_count:.1f} per protocol)")
    
    print("Extraction strategies:")
    for st, cnt in sorted(strategy_counts.items()):
        print(f"  {st:25s}: {cnt:5d}")
        
    avg_len = sum(len(c['content'].split()) for c in all_chunks) / max(len(all_chunks), 1)
    print(f"Avg chunk length: {avg_len:.0f} words")

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
    print("\nEncoding vectors...")
    with ProcessPoolExecutor(max_workers=NUM_GPUS) as ex:
        futures = [ex.submit(encode_on_gpu, i, splits[i]) for i in range(NUM_GPUS)]
        results = [f.result() for f in futures]

    # 5. Merge and save
    print("\nMerging and saving...")
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