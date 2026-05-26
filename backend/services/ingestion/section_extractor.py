import re
from dataclasses import dataclass

from backend.services.ingestion.pdf_parser import ParsedDocument


SECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^abstract\b", re.IGNORECASE | re.MULTILINE), "abstract"),
    (re.compile(r"^\d+\.?\s+introduction\b", re.IGNORECASE | re.MULTILINE), "introduction"),
    (re.compile(r"^\d+\.?\s+related\s+work\b", re.IGNORECASE | re.MULTILINE), "related_work"),
    (re.compile(r"^\d+\.?\s+background\b", re.IGNORECASE | re.MULTILINE), "background"),
    (re.compile(r"^\d+\.?\s+method", re.IGNORECASE | re.MULTILINE), "methods"),
    (re.compile(r"^\d+\.?\s+approach\b", re.IGNORECASE | re.MULTILINE), "methods"),
    (re.compile(r"^\d+\.?\s+experiment", re.IGNORECASE | re.MULTILINE), "experiments"),
    (re.compile(r"^\d+\.?\s+evaluation\b", re.IGNORECASE | re.MULTILINE), "experiments"),
    (re.compile(r"^\d+\.?\s+result", re.IGNORECASE | re.MULTILINE), "results"),
    (re.compile(r"^\d+\.?\s+discussion\b", re.IGNORECASE | re.MULTILINE), "discussion"),
    (re.compile(r"^\d+\.?\s+conclusion", re.IGNORECASE | re.MULTILINE), "conclusion"),
    (re.compile(r"^references\b", re.IGNORECASE | re.MULTILINE), "references"),
    (re.compile(r"^appendix\b", re.IGNORECASE | re.MULTILINE), "appendix"),
]


@dataclass
class Section:
    name: str
    title: str
    text: str
    start_char: int
    start_page: int = 0


class SectionExtractor:
    def extract(self, doc: ParsedDocument) -> list[Section]:
        full_text = doc.full_text
        boundaries: list[tuple[int, str, str]] = []

        for pattern, section_name in SECTION_PATTERNS:
            for match in pattern.finditer(full_text):
                boundaries.append((match.start(), section_name, match.group().strip()))

        if not boundaries:
            return [Section(
                name="body",
                title="Body",
                text=full_text,
                start_char=0,
                start_page=0,
            )]

        boundaries.sort(key=lambda x: x[0])
        sections: list[Section] = []

        # Prepend if text exists before first detected section
        if boundaries[0][0] > 0:
            sections.append(Section(
                name="preamble",
                title="Preamble",
                text=full_text[: boundaries[0][0]].strip(),
                start_char=0,
                start_page=0,
            ))

        for i, (start, name, title) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
            text = full_text[start:end].strip()
            if not text:
                continue
            sections.append(Section(
                name=name,
                title=title,
                text=text,
                start_char=start,
                start_page=self._char_to_page(start, doc),
            ))

        # Skip reference section for retrieval (noise)
        return [s for s in sections if s.name != "references"]

    def _char_to_page(self, char_pos: int, doc: ParsedDocument) -> int:
        cumulative = 0
        for page in doc.pages:
            cumulative += len(page.text)
            if char_pos <= cumulative:
                return page.page_num
        return doc.pages[-1].page_num if doc.pages else 0
