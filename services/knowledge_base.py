import numpy as np
from typing import List, Dict, Any
from services.chunk_service import chunk_text, chunk_text_parent_child, get_text_splitter
from services.embedding_service import generate_embeddings
from services.vector_store import create_vector_store
from services.retriever import Retriever
from services.bm25_retriever import BM25Retriever
from services.models.document import Document
from services.loaders.chunk import Chunk
from config import RETRIEVAL_STRATEGY, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP

class KnowledgeBase:
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, Chunk] = {}
        self.vector_records: List[Dict[str, Any]] = []
        self.index = None
        self.bm25_index = None

    def add_document(self, text: str, metadata: Dict[str, Any]):
        if not text.strip():
            return
        source = metadata.get("source", "unknown")
        title = metadata.get("title", "Document")
        doc = Document.create(source=source, title=title, content=text, metadata=metadata)
        self.documents[doc.id] = doc
        records = []

        if RETRIEVAL_STRATEGY == "parent_child":
            parent_chunks, child_chunks = chunk_text_parent_child(text, doc.id)
            for p_chunk in parent_chunks:
                p_chunk.metadata.update(metadata)
                self.chunks[p_chunk.id] = p_chunk
            for c_chunk in child_chunks:
                c_chunk.metadata.update(metadata)
                self.chunks[c_chunk.id] = c_chunk
                doc.chunk_ids.append(c_chunk.id)
                records.append({"text": c_chunk.text, "metadata": c_chunk.metadata, "chunk": c_chunk})
        else:
            standard_chunks = chunk_text(text, doc.id)
            for chunk in standard_chunks:
                chunk.metadata.update(metadata)
                self.chunks[chunk.id] = chunk
                doc.chunk_ids.append(chunk.id)
                records.append({"text": chunk.text, "metadata": chunk.metadata, "chunk": chunk})

        self._add_records(records)

    def add_chunks(self, chunks_data: List[Dict[str, Any]]):
        if not chunks_data:
            return
        metadata = chunks_data[0].get("metadata", {})
        doc = Document.create(
            source=metadata.get("source", "youtube"),
            title=metadata.get("title", "YouTube Transcript"),
            content=" ".join([c.get("text", "") for c in chunks_data]),
            metadata=metadata
        )
        self.documents[doc.id] = doc
        records = []

        if RETRIEVAL_STRATEGY == "parent_child":
            child_splitter = get_text_splitter(CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP)
            child_idx = 0
            for idx, item in enumerate(chunks_data):
                p_chunk = Chunk(
                    document_id=doc.id,
                    text=item["text"],
                    index_in_doc=idx,
                    metadata=item.get("metadata", {}).copy()
                )
                if "start" in item:
                    p_chunk.metadata["start"] = item["start"]
                if "end" in item:
                    p_chunk.metadata["end"] = item["end"]
                self.chunks[p_chunk.id] = p_chunk

                sub_texts = child_splitter.split_text(item["text"])
                for sub_t in sub_texts:
                    c_chunk = Chunk(
                        document_id=doc.id,
                        parent_chunk_id=p_chunk.id,
                        text=sub_t,
                        index_in_doc=child_idx,
                        metadata=p_chunk.metadata.copy()
                    )
                    self.chunks[c_chunk.id] = c_chunk
                    doc.chunk_ids.append(c_chunk.id)
                    record = {"text": c_chunk.text, "metadata": c_chunk.metadata, "chunk": c_chunk}
                    if "start" in p_chunk.metadata:
                        record["start"] = p_chunk.metadata["start"]
                    if "end" in p_chunk.metadata:
                        record["end"] = p_chunk.metadata["end"]
                    records.append(record)
                    child_idx += 1
        else:
            for idx, item in enumerate(chunks_data):
                chunk = Chunk(
                    document_id=doc.id,
                    text=item["text"],
                    index_in_doc=idx,
                    metadata=item.get("metadata", {}).copy()
                )
                if "start" in item:
                    chunk.metadata["start"] = item["start"]
                if "end" in item:
                    chunk.metadata["end"] = item["end"]
                self.chunks[chunk.id] = chunk
                doc.chunk_ids.append(chunk.id)
                record = {"text": chunk.text, "metadata": chunk.metadata, "chunk": chunk}
                if "start" in item:
                    record["start"] = item["start"]
                if "end" in item:
                    record["end"] = item["end"]
                records.append(record)

        self._add_records(records)

    def _add_records(self, records: List[Dict[str, Any]]):
        texts = [rec["text"] for rec in records]
        embeddings = generate_embeddings(texts)
        embeddings = np.array(embeddings).astype("float32")
        for record, emb in zip(records, embeddings):
            record["chunk"].embedding = emb
        self.vector_records.extend(records)
        if self.index is None:
            self.index = create_vector_store(embeddings)
        else:
            self.index.add(embeddings)
        all_chunks = [rec["chunk"] for rec in self.vector_records]
        self.bm25_index = BM25Retriever(all_chunks)

    def retrieve(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            return []
        retriever = Retriever(
            index=self.index,
            vector_records=self.vector_records,
            chunks_dict=self.chunks,
            bm25_index=self.bm25_index
        )
        return retriever.search(question, top_k)

    def clear(self):
        self.documents.clear()
        self.chunks.clear()
        self.vector_records.clear()
        self.index = None
        self.bm25_index = None
