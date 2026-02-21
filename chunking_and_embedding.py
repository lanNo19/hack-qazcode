import json
import re
import os
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from FlagEmbedding import BGEM3FlagModel
from rank_bm25 import BM25Okapi


def get_structural_chunks_from_dict(data, target_tokens=150):
    protocol_id = data.get('protocol_id', 'unknown')
    icd_codes = data.get('icd_codes', [])
    text = data.get('text', '')
    protocol_title = data.get('source_file', 'Unknown').replace('.pdf', '')

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

    # FIX 1: Fallback so we never get a dangling " > ]" when no headers matched yet
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
        # FIX 2: Return empty lists for sparse (not np.array([])) so merging is consistent
        return {'dense_vecs': np.array([]), 'lexical_weights': []}

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    corpus_texts = [c['content'] for c in chunks]
    outputs = model.encode(corpus_texts, return_dense=True, return_sparse=True)

    # outputs['lexical_weights'] is a list of dicts — return it as-is
    return {
        'dense_vecs': outputs['dense_vecs'],       # np.ndarray (N, dim)
        'lexical_weights': outputs['lexical_weights']  # list of dicts
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
            all_chunks.extend(get_structural_chunks_from_dict(data))

    print(f"Total chunks extracted: {len(all_chunks)}")

    # 2. FIX 3: Build BM25 index for exact-match ICD code retrieval
    print("Building BM25 index...")
    tokenized_corpus = [c['content'].split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    with open('bm25_index.pkl', 'wb') as f:
        pickle.dump(bm25, f)
    print("BM25 index saved to bm25_index.pkl")

    # 3. Split chunks dynamically into NUM_GPUS parts
    k, m = divmod(len(all_chunks), NUM_GPUS)
    chunk_splits = [all_chunks[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(NUM_GPUS)]

    print(f"Distributing chunks across {NUM_GPUS} GPUs...")
    for i, split in enumerate(chunk_splits):
        print(f"GPU {i} will process: {len(split)} chunks")

    # 4. Run parallel processing across all GPUs
    results = []
    with ProcessPoolExecutor(max_workers=NUM_GPUS) as executor:
        futures = [executor.submit(encode_on_gpu, i, chunk_splits[i]) for i in range(NUM_GPUS)]
        for future in futures:
            results.append(future.result())

    # 5. Merge dense embeddings (np.concatenate is fine here — they're real arrays)
    merged_dense = np.concatenate(
        [res['dense_vecs'] for res in results if len(res['dense_vecs']) > 0],
        axis=0
    )

    # FIX 4: Merge sparse weights as a flat list of dicts, NOT np.concatenate
    merged_sparse = []
    for res in results:
        merged_sparse.extend(res['lexical_weights'])

    # 6. Save to disk
    with open('processed_corpus.json', 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    np.savez('embeddings.npz', dense=merged_dense)

    # FIX 5: Sparse weights saved separately as JSON (they're dicts, not arrays)
    with open('sparse_weights.json', 'w', encoding='utf-8') as f:
        json.dump(merged_sparse, f, ensure_ascii=False)

    print(f"Successfully processed and stored {len(all_chunks)} chunks using {NUM_GPUS} GPUs.")
    print(f"  -> embeddings.npz      (dense vectors)")
    print(f"  -> sparse_weights.json (BGE-M3 lexical weights)")
    print(f"  -> bm25_index.pkl      (BM25 for exact ICD code matching)")
    print(f"  -> processed_corpus.json")