import numpy as np

from services.chunk_service import chunk_text
from services.embedding_service import generate_embeddings
from services.vector_store import create_vector_store
from services.retriever import Retriever

class KnowledgeBase:

    def __init__(self):
        self.vector_records = []
        self.index = None

    def add_document(self,text,metadata):
        if not text.strip():
            return
        chunks = chunk_text(text)
        if not chunks:
            return
        records = []
        for chunk in chunks:

         chunk.metadata.update(metadata)

         records.append(
        {
            "text": chunk.text,
            "metadata": chunk.metadata,
            "chunk": chunk
        }
    )
        self._add_records(records)
    def add_chunks(self,chunks):
        if not chunks:
            return
        self._add_records(
            chunks
        )
    def _add_records(self,records):
        texts = []
        for record in records:
            texts.append(
                record["text"]
            )
        embeddings = generate_embeddings(texts)
        embeddings = np.array(embeddings).astype("float32")
        self.vector_records.extend(records)
        if self.index is None:
            self.index = create_vector_store(embeddings)

        else:
            self.index.add(embeddings)

    def retrieve(self,question,top_k=3):
        if self.index is None:
            return []
        retriever = Retriever(
            self.index,
            self.vector_records
        )

        return retriever.search(question,top_k)

    def clear(self):
        self.vector_records.clear()
        self.index = None
