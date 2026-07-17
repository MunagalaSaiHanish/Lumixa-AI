from dataclasses import dataclass, field
from typing import Dict, Optional
import uuid

@dataclass
class Chunk:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: Optional[str] = None
    text: str = ""
    metadata: Dict = field(default_factory=dict)
    embedding = None