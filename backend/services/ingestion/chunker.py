import re

import tiktoken

from backend.data_types import Chunk, ChunkType
from backend.services.ingestion.section_extractor import Section


_EQUATION_RE = re.compile(
    r"\$\$.+?\$\$|\$.+?\$|\\begin\{equation\}|\\begin\{align\}|\\[a-zA-Z]+\{",
    re.DOTALL,
)
_FIGURE_RE = re.compile(r"fig(?:ure)?\.?\s*\d+", re.IGNORECASE)
_TABLE_RE = re.compile(r"table\s+\d+", re.IGNORECASE)


class SemanticChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._enc = tiktoken.get_encoding("cl100k_base")

    def chunk_sections(self, sections: list[Section], paper_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        global_index = 0
        for section in sections:
            section_chunks = self._chunk_text(
                text=section.text,
                paper_id=paper_id,
                section_name=section.name,
                start_page=section.start_page,
                start_index=global_index,
            )
            chunks.extend(section_chunks)
            global_index += len(section_chunks)
        return chunks

    def _chunk_text(
        self,
        text: str,
        paper_id: str,
        section_name: str,
        start_page: int,
        start_index: int,
    ) -> list[Chunk]:
        tokens = self._enc.encode(text)
        chunks: list[Chunk] = []
        pos = 0
        local_idx = 0

        while pos < len(tokens):
            end = min(pos + self.chunk_size, len(tokens))
            chunk_tokens = tokens[pos:end]
            chunk_text = self._enc.decode(chunk_tokens)

            chunks.append(Chunk(
                paper_id=paper_id,
                text=chunk_text,
                chunk_index=start_index + local_idx,
                page=start_page,
                section=section_name,
                chunk_type=self._detect_type(chunk_text),
                token_count=len(chunk_tokens),
            ))

            pos += self.chunk_size - self.overlap
            local_idx += 1

        return chunks

    def _detect_type(self, text: str) -> ChunkType:
        if _EQUATION_RE.search(text):
            return ChunkType.EQUATION
        if _FIGURE_RE.search(text):
            return ChunkType.FIGURE_CAPTION
        if _TABLE_RE.search(text):
            return ChunkType.TABLE
        return ChunkType.TEXT
