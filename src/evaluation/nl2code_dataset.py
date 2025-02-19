import json
import torch
import faiss

import numpy as np

class VectorBase:
    def __init__(self, dimension, record_vectors=False, gpu_id=0):
        self.dimension = dimension
        self.gpu_id = gpu_id
        self.res = faiss.StandardGpuResources()
        self.index = None
        self.vector_urls = []
        if record_vectors:
            self.vectors = []
        self._prepare_index()
    
    def _normalize(self, vectors):
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    def _prepare_index(self):
        index_cpu = faiss.IndexFlatL2(self.dimension)
        self.index = index_cpu
    
    def add(self, vectors, urls):
        normalized_vectors = self._normalize(vectors)
        self.index.add(normalized_vectors)
        self.vector_urls.extend(urls)
        if hasattr(self, 'vectors'):
            for vector in normalized_vectors:
                self.vectors.append(vector)
    
    def search(self, query_vectors, k=5):
        normalized_queries = self._normalize(query_vectors)
        distances, indices = self.index.search(normalized_queries, k)
        cosine_similarities = 1 - 0.5 * distances
        return cosine_similarities, indices
    
    def calculate_mrr_at_k(self, query_urls, indices, max_k):
        mrr_scores = {k: 0.0 for k in range(1, max_k + 1)}
        for i, query_url in enumerate(query_urls):
            found = False
            for rank, idx in enumerate(indices[i], start=1):
                if self.vector_urls[idx] == query_url:
                    for k in range(rank, max_k + 1):
                        mrr_scores[k] += 1.0 / rank
                    found = True
                    break
            if not found:
                for k in range(1, max_k + 1):
                    mrr_scores[k] += 0.0 

        num_queries = len(query_urls)
        mrr_scores = {k: mrr_scores[k] / num_queries for k in mrr_scores}
        return mrr_scores
    
    def calculate_mrr_at_k_for_hard_cases(self, query_urls, indices, hard_cases, max_k):
        mrr_scores = {k: 0.0 for k in range(1, max_k + 1)}
        for i, query_url in enumerate(query_urls):
            if query_url not in hard_cases:
                continue
            found = False
            for rank, idx in enumerate(indices[i], start=1):
                if self.vector_urls[idx] == query_url:
                    for k in range(rank, max_k + 1):
                        mrr_scores[k] += 1.0 / rank
                    found = True
                    break
            if not found:
                for k in range(1, max_k + 1):
                    mrr_scores[k] += 0.0 

        num_queries = len(hard_cases)
        mrr_scores = {k: mrr_scores[k] / num_queries for k in mrr_scores}
        return mrr_scores
    
    def calculate_hit_rate_at_k(self, query_urls, indices, max_k):
        hit_rate_scores = {k: 0.0 for k in range(1, max_k + 1)}
        for i, query_url in enumerate(query_urls):
            for rank, idx in enumerate(indices[i], start=1):
                if self.vector_urls[idx] == query_url:
                    for k in range(rank, max_k + 1):
                        hit_rate_scores[k] += 1.0
                    break

        num_queries = len(query_urls)
        hit_rate_scores = {k: hit_rate_scores[k] / num_queries for k in hit_rate_scores}
        return hit_rate_scores