from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import uuid

@dataclass
class Chunk:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    index_in_doc: int = 0
    embedding: Optional[Any] = None