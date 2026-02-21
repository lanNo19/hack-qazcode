import json
import numpy as np
import os
from FlagEmbedding import BGEM3FlagModel, FlagReranker
import torch

class MedicalRetriever:
    def __init__(self, corpus_path, embed_path):
        # Initialize models. 
        # Adding 'devices' explicitly can sometimes bypass unnecessary multiprocessing
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading models on {device}...")
        
        # devices=['cuda:0'] или ['cpu'] принудительно отключает лишнюю многопроцессорность
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
        self.dense_embeddings = data['dense']
        self.sparse_embeddings = data['sparse']

    def compute_sparse_score(self, query_sparse, doc_sparse):
        score = 0
        for token, weight in query_sparse.items():
            if token in doc_sparse:
                score += weight * doc_sparse[token]
        return score

    def retrieve(self, query_text, top_k=20, rerank_top_n=5):
        # The .encode() call here is what triggered your error
        q_out = self.embed_model.encode([query_text], return_dense=True, return_sparse=True)
        q_dense = q_out['dense_vecs'][0]
        q_sparse = q_out['lexical_weights'][0]

        scores = []
        for i in range(len(self.corpus)):
            d_score = np.dot(q_dense, self.dense_embeddings[i])
            s_score = self.compute_sparse_score(q_sparse, self.sparse_embeddings[i])
            hybrid_score = (0.7 * d_score) + (0.3 * s_score)
            scores.append((i, hybrid_score))

        top_indices = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
        candidates = [self.corpus[i] for i, _ in top_indices]

        rerank_pairs = [[query_text, cand['content']] for cand in candidates]
        rerank_scores = self.reranker.compute_score(rerank_pairs)
        
        for i, score in enumerate(rerank_scores):
            candidates[i]['rerank_score'] = score
        
        return sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)[:rerank_top_n]

# --- EVERYTHING BELOW MUST BE INSIDE THIS BLOCK ---
if __name__ == "__main__":
    # 1. Configuration
    CORPUS_FILE = 'processed_corpus.json'
    EMBED_FILE = 'embeddings.npz'
    
    if not os.path.exists(CORPUS_FILE) or not os.path.exists(EMBED_FILE):
        print(f"Error: Could not find {CORPUS_FILE} or {EMBED_FILE}. Please run your embedding script first.")
    else:
        # 2. Initialize Retriever
        print("Initializing models and loading data...")
        retriever = MedicalRetriever(CORPUS_FILE, EMBED_FILE)
        
        # 3. Test Query
        user_symptoms = "Острая боль в правом подреберье, тошнота, срок беременности 32 недели"
        print(f"\nQuery: {user_symptoms}")
        
        # 4. Run Retrieval
        results = retriever.retrieve(user_symptoms)

        # 5. Display Results
        print("\nTop Diagnoses:")
        for i, res in enumerate(results):
            print(f"{i+1}. ICD-10: {res['metadata'].get('icd_codes', 'N/A')}")
            print(f"   Score: {res['rerank_score']:.4f}")
            print(f"   Text: {res['content'][:150]}...\n")