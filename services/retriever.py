import numpy as np
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import config
from services.embedding_service import model
from services.llm_service import generate_alternative_queries
from services.loaders.chunk import Chunk

_reranker_model = None
_reranker_model_name = None

def get_reranker(model_name: str) -> CrossEncoder:
    """Lazy load Cross-Encoder"""
    global _reranker_model, _reranker_model_name
    if _reranker_model is None or _reranker_model_name != model_name:
        _reranker_model = CrossEncoder(model_name)
        _reranker_model_name = model_name
    return _reranker_model

class Retriever:
    def __init__(self, index, vector_records: List[Dict[str, Any]], chunks_dict: Dict[str, Chunk] = None, bm25_index = None):
        self.index = index
        self.vector_records = vector_records
        self.chunks_dict = chunks_dict or {}
        self.bm25_index = bm25_index

    def _preprocess_query(self, query: str) -> str:
        return query.strip().lower()

    def search(self, question: str, top_k: int = None) -> List[Dict[str, Any]]:
        """RAG search pipeline"""
        if self.index is None or not question.strip():
            return []

        target_top_k = top_k if top_k is not None else config.TOP_K
        enable_hybrid = config.ENABLE_HYBRID_SEARCH
        rrf_k = config.RRF_K
        enable_expansion = config.ENABLE_QUERY_EXPANSION
        enable_reranking = config.ENABLE_RERANKING
        reranker_model_name = config.RERANKER_MODEL
        strategy = config.RETRIEVAL_STRATEGY
        neighbor_window = config.NEIGHBOR_WINDOW

        clean_query = self._preprocess_query(question)
        queries_to_search = [clean_query]
        if enable_expansion:
            alternative_queries = generate_alternative_queries(question)
            for alt in alternative_queries:
                alt_clean = self._preprocess_query(alt)
                if alt_clean and alt_clean not in queries_to_search:
                    queries_to_search.append(alt_clean)

        all_vector_matches = []
        all_bm25_matches = []

        for q in queries_to_search:
            q_emb = model.encode([q])
            q_emb = np.array(q_emb).astype("float32")
            search_k = max(target_top_k * 3, 20)
            distances, indices = self.index.search(q_emb, search_k)
            
            vec_hits = []
            for idx in indices[0]:
                if idx == -1 or idx >= len(self.vector_records):
                    continue
                vec_hits.append(self.vector_records[idx])
            all_vector_matches.append(vec_hits)

            if enable_hybrid and self.bm25_index is not None:
                bm25_hits_with_scores = self.bm25_index.search(q, search_k)
                bm25_hits = []
                for chunk_obj, score in bm25_hits_with_scores:
                    for record in self.vector_records:
                        record_chunk = record.get("chunk")
                        if record_chunk and record_chunk.id == chunk_obj.id:
                            bm25_hits.append(record)
                            break
                all_bm25_matches.append(bm25_hits)

        rrf_scores = {}
        def update_ranks(matches_list, key_name):
            for matches in matches_list:
                for rank, record in enumerate(matches, start=1):
                    chunk_obj = record.get("chunk")
                    if not chunk_obj:
                        continue
                    chunk_id = chunk_obj.id
                    if chunk_id not in rrf_scores:
                        rrf_scores[chunk_id] = {"record": record, "vector_rank": 9999, "bm25_rank": 9999}
                    rrf_scores[chunk_id][key_name] = min(rrf_scores[chunk_id][key_name], rank)

        update_ranks(all_vector_matches, "vector_rank")
        if enable_hybrid and self.bm25_index is not None:
            update_ranks(all_bm25_matches, "bm25_rank")

        candidates = []
        for chunk_id, ranks in rrf_scores.items():
            score = 0.0
            if ranks["vector_rank"] != 9999:
                score += 1.0 / (rrf_k + ranks["vector_rank"])
            if ranks["bm25_rank"] != 9999:
                score += 1.0 / (rrf_k + ranks["bm25_rank"])
            candidates.append((ranks["record"], score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [item[0] for item in candidates[:max(target_top_k * 2, 10)]]

        final_retrieved = []
        if enable_reranking and top_candidates:
            try:
                rerank_model = get_reranker(reranker_model_name)
                pairs = [(question, record["text"]) for record in top_candidates]
                scores = rerank_model.predict(pairs)
                ranked_candidates = sorted(zip(top_candidates, scores), key=lambda x: x[1], reverse=True)
                final_retrieved = [item[0] for item in ranked_candidates[:target_top_k]]
            except Exception as e:
                print(f"Reranking error: {e}. Fallback to RRF.")
                final_retrieved = top_candidates[:target_top_k]
        else:
            final_retrieved = top_candidates[:target_top_k]

        expanded_results = []
        seen_texts = set()

        for record in final_retrieved:
            chunk_obj = record.get("chunk")
            if not chunk_obj:
                expanded_results.append(record)
                continue

            result_text = record["text"]
            result_metadata = record.get("metadata", {}).copy()

            if strategy == "parent_child":
                parent_id = chunk_obj.parent_chunk_id
                if parent_id and parent_id in self.chunks_dict:
                    parent_chunk = self.chunks_dict[parent_id]
                    result_text = parent_chunk.text
                    result_metadata.update(parent_chunk.metadata)
            elif strategy == "neighbor":
                doc_id = chunk_obj.document_id
                if doc_id:
                    curr_idx = chunk_obj.index_in_doc
                    window = neighbor_window
                    siblings = []
                    for cid, c in self.chunks_dict.items():
                        if c.document_id == doc_id and abs(c.index_in_doc - curr_idx) <= window:
                            siblings.append(c)
                    siblings.sort(key=lambda x: x.index_in_doc)
                    result_text = "\n[...]\n".join([sib.text for sib in siblings])
                    for sib in siblings:
                        if sib.metadata:
                            result_metadata.update(sib.metadata)

            if result_text.strip() in seen_texts:
                continue
            seen_texts.add(result_text.strip())

            new_record = {
                "text": result_text,
                "metadata": result_metadata,
                "chunk": chunk_obj
            }
            if "start" in record:
                new_record["start"] = record["start"]
            elif "start" in result_metadata:
                new_record["start"] = result_metadata["start"]
            if "end" in record:
                new_record["end"] = record["end"]
            elif "end" in result_metadata:
                new_record["end"] = result_metadata["end"]

            expanded_results.append(new_record)

        return expanded_results
