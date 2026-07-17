from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
from loaders.chunk import Chunk


SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS
    )

def chunk_text(text):
    if not text.strip():
        return []
    text_splitter = get_text_splitter()
    texts = text_splitter.split_text(text)

    return [
    Chunk(text=chunk)
    for chunk in texts
    ]

def chunk_transcript(transcript_segments):
    text_splitter = get_text_splitter()
    chunks = []
    current_text = ""
    current_start = None
    current_end = None
    for segment in transcript_segments:
        if current_start is None:
            current_start = segment["start"]
        current_text += segment["text"] + " "
        current_end = segment["end"]
        if len(current_text) >= CHUNK_SIZE:
            split_chunks = text_splitter.split_text(current_text)
            for chunk in split_chunks:
                chunks.append({
                    "text": chunk,
                    "start": current_start,
                    "end": current_end
                })
            current_text = ""
            current_start = None
            current_end = None
    if current_text:
        split_chunks = text_splitter.split_text(current_text)
        for chunk in split_chunks:
            chunks.append({
                "text": chunk,
                "start": current_start,
                "end": current_end
            })
    return chunks

def chunk_documents(documents):
    text_splitter = get_text_splitter()
    chunks = []
    for document in documents:
        split_chunks = text_splitter.split_text(document["text"])
        for chunk in split_chunks:
            chunks.append({
                "text": chunk,
                "metadata": document["metadata"]
            })
    return chunks
