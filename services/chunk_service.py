from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP
)
from services.loaders.chunk import Chunk

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

def get_text_splitter(size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, separators=SEPARATORS)

def chunk_text(text: str, document_id: str = None) -> list[Chunk]:
    if not text.strip():
        return []
    splitter = get_text_splitter(CHUNK_SIZE, CHUNK_OVERLAP)
    split_texts = splitter.split_text(text)
    chunks = []
    for idx, text_block in enumerate(split_texts):
        chunks.append(Chunk(document_id=document_id, text=text_block, index_in_doc=idx))
    return chunks

def chunk_text_parent_child(text: str, document_id: str = None) -> tuple[list[Chunk], list[Chunk]]:
    if not text.strip():
        return [], []
    parent_splitter = get_text_splitter(PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP)
    child_splitter = get_text_splitter(CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP)
    parent_texts = parent_splitter.split_text(text)
    parent_chunks = []
    child_chunks = []
    child_index = 0
    for p_idx, p_text in enumerate(parent_texts):
        parent_chunk = Chunk(document_id=document_id, text=p_text, index_in_doc=p_idx)
        parent_chunks.append(parent_chunk)
        child_texts = child_splitter.split_text(p_text)
        for c_text in child_texts:
            child_chunk = Chunk(
                document_id=document_id,
                parent_chunk_id=parent_chunk.id,
                text=c_text,
                index_in_doc=child_index
            )
            child_chunks.append(child_chunk)
            child_index += 1
    return parent_chunks, child_chunks

def chunk_transcript(transcript_segments: list[dict], document_id: str = None) -> list[dict]:
    splitter = get_text_splitter(CHUNK_SIZE, CHUNK_OVERLAP)
    chunks_data = []
    current_text = ""
    current_start = None
    current_end = None
    for segment in transcript_segments:
        if current_start is None:
            current_start = segment["start"]
        current_text += segment["text"] + " "
        current_end = segment["end"]
        if len(current_text) >= CHUNK_SIZE:
            split_chunks = splitter.split_text(current_text)
            for chunk in split_chunks:
                chunks_data.append({"text": chunk, "start": current_start, "end": current_end})
            current_text = ""
            current_start = None
            current_end = None
    if current_text:
        split_chunks = splitter.split_text(current_text)
        for chunk in split_chunks:
            chunks_data.append({"text": chunk, "start": current_start, "end": current_end})
    return chunks_data

def chunk_transcript_parent_child(transcript_segments: list[dict], document_id: str = None) -> tuple[list[Chunk], list[Chunk]]:
    parent_splitter = get_text_splitter(PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP)
    child_splitter = get_text_splitter(CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP)
    parent_blocks = []
    current_text = ""
    current_start = None
    current_end = None
    for segment in transcript_segments:
        if current_start is None:
            current_start = segment["start"]
        current_text += segment["text"] + " "
        current_end = segment["end"]
        if len(current_text) >= PARENT_CHUNK_SIZE:
            split_texts = parent_splitter.split_text(current_text)
            for txt in split_texts:
                parent_blocks.append({"text": txt, "start": current_start, "end": current_end})
            current_text = ""
            current_start = None
            current_end = None
    if current_text:
        split_texts = parent_splitter.split_text(current_text)
        for txt in split_texts:
            parent_blocks.append({"text": txt, "start": current_start, "end": current_end})
    parent_chunks = []
    child_chunks = []
    child_index = 0
    for p_idx, block in enumerate(parent_blocks):
        parent_chunk = Chunk(
            document_id=document_id,
            text=block["text"],
            index_in_doc=p_idx,
            metadata={"start": block["start"], "end": block["end"]}
        )
        parent_chunks.append(parent_chunk)
        child_texts = child_splitter.split_text(block["text"])
        for c_text in child_texts:
            child_chunk = Chunk(
                document_id=document_id,
                parent_chunk_id=parent_chunk.id,
                text=c_text,
                index_in_doc=child_index,
                metadata={"start": block["start"], "end": block["end"]}
            )
            child_chunks.append(child_chunk)
            child_index += 1
    return parent_chunks, child_chunks
