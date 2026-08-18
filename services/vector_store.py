import faiss
import numpy as np
from services.embedding_service import model

# Create FAISS Vector Store
def create_vector_store(embeddings):
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Retrieve Documents
def retrieve_documents(question, index, vector_records, top_k=3):
    if index is None:
        return []
    question_embedding = model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")
    distances, indices = index.search(question_embedding, top_k)

    retrieved_documents = []
    for idx in indices[0]:
        # FAISS may return -1 when nothing is found
        if idx == -1:
            continue
        if idx >= len(vector_records):
            continue
        retrieved_documents.append(vector_records[idx])

    return retrieved_documents
