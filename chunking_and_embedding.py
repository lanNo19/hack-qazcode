import json
import re
import os
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from FlagEmbedding import BGEM3FlagModel
from rank_bm25 import BM25Okapi


# Matches Roman-numeral section headers embedded in continuous text,
# e.g. "... I. ВВОДНАЯ ЧАСТЬ ..." or "... II. ОПРЕДЕЛЕНИЕ ..."
_SECTION_SPLIT_RE = re.compile(r'(?=\s[IVXLC]+\.\s+[А-Я][А-Я\s]{2,})')

# Boilerplate ends and real content begins at the first Roman-numeral section.
# We detect this by finding "I." followed by Cyrillic uppercase words.
_BOILERPLATE_END_RE = re.compile(r'\bI\.\s+[А-Я][А-Я\s]{2,}')


def get_structural_chunks_from_dict(data, target_tokens=150):
    protocol_id = data.get('protocol_id', 'unknown')
    icd_codes = data.get('icd_codes', [])
    text = data.get('text', '')
    protocol_title = data.get('source_file', 'Unknown').replace('.pdf', '')

    # --- Strip boilerplate ---
    # The PDF parser produces one long string. The approval block always comes
    # before "I. ВВОДНАЯ ЧАСТЬ" (or equivalent first Roman-numeral section).
    # Find where actual content starts and discard everything before it.
    match = _BOILERPLATE_END_RE.search(text)
    if match:
        text = text[match.start():]

    # --- Normalize into lines ---
    # Split the continuous text at Roman-numeral section boundaries so the
    # downstream header-matching logic has something to work with.
    # Also split on numbered subsections like "1.1 " and "1.2 " etc.
    # Replace those boundaries with a newline + the matched text.
    text = re.sub(r'(\s)([IVXLC]+\.\s+[А-Я])', r'\n\2', text)
    text = re.sub(r'(\s)(\d+\.\d+\s+[А-Яа-я])', r'\n\2', text)
    text = re.sub(r'(\s)(\d+\.\s+[А-Яа-я])', r'\n\2', text)

    header_patterns = [
        r'^([IVXLC]+\.\s+[А-Я\s]+)$',
        r'^(\d+\.\s+[А-Яа-я\s,]+)$',
        r'^(\d+\.\d+\s+[А-Яа-я\s\(\),]+)$'
    ]

    lines = text.split('\n')
    structured_chunks = []
    current_headers = ["", "", ""]
    current_chunk_text = []
    current_tokens = 0

    def get_context_string():
        active_headers = [h for h in current_headers if h]
        if active_headers:
            return f"[{protocol_title} > " + " > ".join(active_headers) + "]"
        return f"[{protocol_title}]"

    for line in lines:
        line = line.strip()
        if not line:
            continue

        is_header = False
        for i, pattern in enumerate(header_patterns):
            if re.match(pattern, line):
                current_headers[i] = line
                for j in range(i + 1, len(current_headers)):
                    current_headers[j] = ""
                is_header = True
                break

        if is_header:
            continue

        line_tokens = len(line.split())
        if current_tokens + line_tokens > target_tokens and current_chunk_text:
            structured_chunks.append({
                "content": get_context_string() + "\n" + "\n".join(current_chunk_text),
                "metadata": {
                    "protocol_id": protocol_id,
                    "icd_codes": icd_codes,
                    "section": current_headers[1] or current_headers[0]
                }
            })
            current_chunk_text, current_tokens = [], 0

        current_chunk_text.append(line)
        current_tokens += line_tokens

    if current_chunk_text:
        structured_chunks.append({
            "content": get_context_string() + "\n" + "\n".join(current_chunk_text),
            "metadata": {"protocol_id": protocol_id, "icd_codes": icd_codes}
        })

    return structured_chunks


def encode_on_gpu(gpu_id, chunks):
    """Loads the model on a specific GPU and processes a subset of chunks."""
    if not chunks:
        return {'dense_vecs': np.array([]), 'lexical_weights': []}

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    corpus_texts = [c['content'] for c in chunks]
    outputs = model.encode(corpus_texts, return_dense=True, return_sparse=True)

    return {
        'dense_vecs': outputs['dense_vecs'],
        'lexical_weights': outputs['lexical_weights']
    }


if __name__ == '__main__':
    NUM_GPUS = 5  # <--- Change this if your hardware setup changes

    # 1. Parse JSONL
    all_chunks = []
    jsonl_file_path = 'corpus/protocols_corpus.jsonl'

    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            chunks = get_structural_chunks_from_dict(data)
            all_chunks.extend(chunks)

    print(f"Total chunks extracted: {len(all_chunks)}")

    # Sanity check: print the first chunk to verify boilerplate is gone
    if all_chunks:
        print("\nFirst chunk preview (should be clinical content, not boilerplate):")
        print(all_chunks[0]['content'][:300])
        print()

    # 2. Build BM25 index
    print("Building BM25 index...")
    tokenized_corpus = [c['content'].split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    with open('bm25_index.pkl', 'wb') as f:
        pickle.dump(bm25, f)
    print("BM25 index saved to bm25_index.pkl")

    # 3. Split chunks across GPUs
    k, m = divmod(len(all_chunks), NUM_GPUS)
    chunk_splits = [all_chunks[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(NUM_GPUS)]

    print(f"Distributing chunks across {NUM_GPUS} GPUs...")
    for i, split in enumerate(chunk_splits):
        print(f"GPU {i} will process: {len(split)} chunks")

    # 4. Run parallel encoding
    results = []
    with ProcessPoolExecutor(max_workers=NUM_GPUS) as executor:
        futures = [executor.submit(encode_on_gpu, i, chunk_splits[i]) for i in range(NUM_GPUS)]
        for future in futures:
            results.append(future.result())

    # 5. Merge dense embeddings
    merged_dense = np.concatenate(
        [res['dense_vecs'] for res in results if len(res['dense_vecs']) > 0],
        axis=0
    )

    # 6. Merge sparse weights (list of dicts, not arrays)
    merged_sparse = []
    for res in results:
        merged_sparse.extend(res['lexical_weights'])

    # 7. Save to disk
    with open('processed_corpus.json', 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    np.savez('embeddings.npz', dense=merged_dense)

    merged_sparse_serializable = [
        {token: float(weight) for token, weight in doc.items()}
        for doc in merged_sparse
    ]
    with open('sparse_weights.json', 'w', encoding='utf-8') as f:
        json.dump(merged_sparse_serializable, f, ensure_ascii=False)

    print(f"\nSuccessfully processed and stored {len(all_chunks)} chunks using {NUM_GPUS} GPUs.")
    print(f"  -> embeddings.npz")
    print(f"  -> sparse_weights.json")
    print(f"  -> bm25_index.pkl")
    print(f"  -> processed_corpus.json")