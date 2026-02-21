import json
import pickle
import numpy as np
import os
from FlagEmbedding import BGEM3FlagModel, FlagReranker
import torch


class MedicalRetriever:
    def __init__(self, corpus_path, embed_path, sparse_path, bm25_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading models on {device}...")

        self.embed_model = BGEM3FlagModel(
            'BAAI/bge-m3',
            use_fp16=(device == "cuda"),
            devices=[device]
        )

        self.reranker = FlagReranker(
            'BAAI/bge-reranker-v2-m3',
            use_fp16=(device == "cuda"),
            devices=[device]
        )

        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.corpus = json.load(f)

        data = np.load(embed_path, allow_pickle=True)
        self.dense_embeddings = data['dense']  # shape: (N, dim)

        with open(sparse_path, 'r', encoding='utf-8') as f:
            self.sparse_embeddings = json.load(f)  # list of dicts: token -> float

        with open(bm25_path, 'rb') as f:
            self.bm25 = pickle.load(f)

        print(f"Loaded corpus: {len(self.corpus)} chunks")

    def compute_sparse_score(self, query_sparse, doc_sparse):
        """Dot product over shared tokens between query and doc lexical weight dicts."""
        score = 0.0
        for token, weight in query_sparse.items():
            if token in doc_sparse:
                score += weight * doc_sparse[token]
        return score

    def retrieve(self, query_text, top_k=20, rerank_top_n=5,
                 dense_weight=0.5, sparse_weight=0.3, bm25_weight=0.2):
        """
        Three-way hybrid retrieval:
          - Dense (BGE-M3):  semantic symptom-to-disease matching
          - Sparse (BGE-M3): subword lexical overlap
          - BM25:            exact ICD code / alphanumeric term matching
        Then cross-encoder reranking on the top_k candidates.
        """
        q_out = self.embed_model.encode([query_text], return_dense=True, return_sparse=True)
        q_dense = q_out['dense_vecs'][0]        # shape: (dim,)
        q_sparse = q_out['lexical_weights'][0]   # dict: token -> weight

        # Vectorised dense scoring over the full corpus
        dense_scores = self.dense_embeddings @ q_dense  # shape: (N,)

        # BM25 scores, normalised to [0, 1]
        bm25_scores = np.array(self.bm25.get_scores(query_text.split()))
        bm25_max = bm25_scores.max()
        if bm25_max > 0:
            bm25_scores = bm25_scores / bm25_max

        # Pre-filter to dense top-100 before computing sparse scores
        # (sparse scoring is a Python dict loop — too slow over the full corpus)
        dense_top100_indices = np.argpartition(dense_scores, -100)[-100:]

        scores = []
        for i in dense_top100_indices:
            s_score = self.compute_sparse_score(q_sparse, self.sparse_embeddings[i])
            hybrid_score = (
                dense_weight  * float(dense_scores[i]) +
                sparse_weight * s_score +
                bm25_weight   * float(bm25_scores[i])
            )
            scores.append((int(i), hybrid_score))

        # Deduplicate by protocol: keep only the best-scoring chunk per protocol_id.
        # Boilerplate is gone (stripped at chunking time), so the best chunk is now
        # genuinely the most relevant clinical content chunk for this protocol.
        seen_protocols = {}
        for idx, score in sorted(scores, key=lambda x: x[1], reverse=True)[:top_k * 3]:
            pid = self.corpus[idx]['metadata'].get('protocol_id', idx)
            if pid not in seen_protocols or score > seen_protocols[pid][1]:
                seen_protocols[pid] = (idx, score)

        # Re-sort deduped results, drop protocols with no ICD codes, take top_k
        deduped = sorted(seen_protocols.values(), key=lambda x: x[1], reverse=True)[:top_k]
        candidates = [
            self.corpus[idx] for idx, _ in deduped
            if self.corpus[idx]['metadata'].get('icd_codes')
        ]

        # Cross-encoder reranking
        rerank_pairs = [[query_text, cand['content']] for cand in candidates]
        rerank_scores = self.reranker.compute_score(rerank_pairs)

        for i, score in enumerate(rerank_scores):
            candidates[i]['rerank_score'] = score

        return sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)[:rerank_top_n]


if __name__ == "__main__":
    CORPUS_FILE = 'processed_corpus.json'
    EMBED_FILE  = 'embeddings.npz'
    SPARSE_FILE = 'sparse_weights.json'
    BM25_FILE   = 'bm25_index.pkl'

    missing = [f for f in [CORPUS_FILE, EMBED_FILE, SPARSE_FILE, BM25_FILE]
               if not os.path.exists(f)]
    if missing:
        print(f"Error: Missing files: {missing}. Please run chunking_and_embedding.py first.")
    else:
        print("Initializing models and loading data...")
        retriever = MedicalRetriever(CORPUS_FILE, EMBED_FILE, SPARSE_FILE, BM25_FILE)

        user_symptoms = "Острая боль в правом подреберье, тошнота, срок беременности 32 недели"
        print(f"\nQuery: {user_symptoms}")

        results = retriever.retrieve(user_symptoms)

        print("\nTop Diagnoses:")
        for i, res in enumerate(results):
            print(f"{i+1}. ICD-10: {res['metadata'].get('icd_codes', 'N/A')}")
            print(f"   Score: {res['rerank_score']:.4f}")
            print(f"   Text: {res['content'][:150]}...\n")