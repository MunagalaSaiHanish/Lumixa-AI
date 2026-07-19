from dataclasses import dataclass, field
from typing import Dict, List, Any
import uuid

@dataclass
class Document:
    id: str
    source: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_ids: List[str] = field(default_factory=list)

    @staticmethod
    def create(source: str, title: str, content: str, metadata: Dict[str, Any] = None) -> "Document":
        if metadata is None:
            metadata = {}
        return Document(
            id=str(uuid.uuid4()),
            source=source,
            title=title,
            content=content,
            metadata=metadata,
            chunk_ids=[]
        )
