import json
import re
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from FlagEmbedding import BGEM3FlagModel

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

    def get_context_string():
        active_headers = [h for h in current_headers if h]
        return f"[{protocol_title} > " + " > ".join(active_headers) + "]"

    for line in lines:
        line = line.strip()
        if not line: continue

        is_header = False
        for i, pattern in enumerate(header_patterns):
            if re.match(pattern, line):
                current_headers[i] = line
                for j in range(i + 1, len(current_headers)): current_headers[j] = ""
                is_header = True
                break

        if is_header: continue

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
    # If a chunk list happens to be empty (e.g., fewer chunks than GPUs), skip gracefully
    if not chunks:
        return {'dense_vecs': np.array([]), 'lexical_weights': np.array([])}

    # Hide the other GPUs from this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Load Model onto the assigned GPU
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    
    corpus_texts = [c['content'] for c in chunks]
    outputs = model.encode(corpus_texts, return_dense=True, return_sparse=True)
    
    return outputs


if __name__ == '__main__':
    NUM_GPUS = 5  # <--- Change this if your hardware setup changes
    
    # 1. Parse JSONL
    all_chunks = []
    jsonl_file_path = 'corpus/protocols_corpus.jsonl'

    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            data = json.loads(line)
            all_chunks.extend(get_structural_chunks_from_dict(data))

    print(f"Total chunks extracted: {len(all_chunks)}")

    # 2. Split chunks dynamically into NUM_GPUS parts
    # This math safely divides the list into N almost-equal sized lists
    k, m = divmod(len(all_chunks), NUM_GPUS)
    chunk_splits = [all_chunks[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(NUM_GPUS)]

    print(f"Distributing chunks across {NUM_GPUS} GPUs...")
    for i, split in enumerate(chunk_splits):
        print(f"GPU {i} will process: {len(split)} chunks")

    # 3. Run parallel processing across all GPUs
    results = []
    with ProcessPoolExecutor(max_workers=NUM_GPUS) as executor:
        # Submit tasks in a loop
        futures = [executor.submit(encode_on_gpu, i, chunk_splits[i]) for i in range(NUM_GPUS)]
        
        # Collect results in order
        for future in futures:
            results.append(future.result())

    # 4. Merge results back together in the correct order
    merged_dense = np.concatenate([res['dense_vecs'] for res in results if len(res['dense_vecs']) > 0], axis=0)
    merged_sparse = np.concatenate([res['lexical_weights'] for res in results if len(res['lexical_weights']) > 0], axis=0)

    # 5. Save to disk
    with open('processed_corpus.json', 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    np.savez('embeddings.npz',
             dense=merged_dense,
             sparse=merged_sparse)

    print(f"Successfully processed and stored {len(all_chunks)} chunks using {NUM_GPUS} GPUs.")