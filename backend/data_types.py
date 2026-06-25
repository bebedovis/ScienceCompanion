import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class ChunkType(str, Enum): 
    TEXT= "text"
    EQUATION = "equation"
    FIGURE_CAPTION = "figure_caption"
    TABLE = "table"
    ABSTRACT = "abstract"

class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"

@dataclass
class DocumentMetadata:
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    abstract: str = ""
    subject_tags: list[str] = field(default_factory=list)
    journal: Optional[str] = None
    arxiv_id: str = ""

@dataclass
class Document:
    id: str = ""
    filename: str = ""
    file_path: str = ""
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    chunk_count: int = 0

@dataclass
class Chunk:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    paper_id: str = ""
    text: str = ""
    chunk_index: int = 0
    page: int = 0
    section: str = "unknown"
    chunk_type: ChunkType = ChunkType.TEXT
    token_count: int = 0
    extra_metadata: dict = field(default_factory=dict)
    def to_metadata(self, doc: Document) -> dict:
        return {
            "paper_id": self.paper_id,
            "section": self.section,
            "page": self.page,
            "chunk_type": self.chunk_type.value,
            "title": doc.metadata.title,
            "authors": ", ".join(doc.metadata.authors),
            "year": doc.metadata.year or 0,
            "journal": doc.metadata.journal or ""
        }
