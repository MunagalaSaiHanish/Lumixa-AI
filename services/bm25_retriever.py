import math
import re
from typing import List, Dict, Any

class BM25Retriever:
    def __init__(self, corpus_chunks: List[Any], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus_chunks
        self.doc_count = len(corpus_chunks)
        self.tokenized_corpus = [self._tokenize(chunk.text) for chunk in corpus_chunks]
        self.doc_lengths = [len(tokens) for tokens in self.tokenized_corpus]
        self.avg_doc_length = sum(self.doc_lengths) / max(self.doc_count, 1)
        self.doc_frequencies: Dict[str, int] = {}
        for tokens in self.tokenized_corpus:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_frequencies[token] = self.doc_frequencies.get(token, 0) + 1
        self.idf: Dict[str, float] = {}
        for word, freq in self.doc_frequencies.items():
            self.idf[word] = math.log(1.0 + (self.doc_count - freq + 0.5) / (freq + 0.5))

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def search(self, query: str, top_k: int = 5) -> List[tuple[Any, float]]:
        if not query.strip() or self.doc_count == 0:
            return []
        query_tokens = self._tokenize(query)
        scores = []
        for idx in range(self.doc_count):
            score = 0.0
            doc_tokens = self.tokenized_corpus[idx]
            doc_len = self.doc_lengths[idx]
            token_counts: Dict[str, int] = {}
            for token in doc_tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
            for q_term in query_tokens:
                if q_term not in token_counts:
                    continue
                tf = token_counts[q_term]
                idf_val = self.idf.get(q_term, 0.0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(self.avg_doc_length, 0.0001)))
                score += idf_val * (numerator / denominator)
            scores.append((self.corpus[idx], score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
