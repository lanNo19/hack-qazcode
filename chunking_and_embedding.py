import json
import re
import numpy as np
from FlagEmbedding import BGEM3FlagModel


# Adjusted to accept a dictionary directly instead of a file path
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


# --- PROCESSING AND STORING ---
# 1. Load Model (ensure local path for competition)
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

all_chunks = []
jsonl_file_path = 'corpus/protocols_corpus.jsonl'

# Read the JSONL file line by line
with open(jsonl_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # Parse the JSON string into a Python dictionary
        data = json.loads(line)

        # Process the dictionary and append chunks
        all_chunks.extend(get_structural_chunks_from_dict(data))

# 2. Generate Hybrid Embeddings
# BGE-M3 generates both Dense and Sparse in one pass
corpus_texts = [c['content'] for c in all_chunks]
outputs = model.encode(corpus_texts, return_dense=True, return_sparse=True)

# 3. Save to disk for Inference
# We save chunks as JSON and embeddings as NPZ/Pickle
with open('processed_corpus.json', 'w', encoding='utf-8') as f:
    json.dump(all_chunks, f, ensure_ascii=False)

np.savez('embeddings.npz',
         dense=outputs['dense_vecs'],
         sparse=np.array(outputs['lexical_weights'], dtype=object))

print(f"Stored {len(all_chunks)} chunks with dense and sparse vectors.")