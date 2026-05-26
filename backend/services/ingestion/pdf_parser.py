from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF


@dataclass
class TextBlock:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    block_type: int  # 0 = text, 1 = image


@dataclass
class PageContent:
    page_num: int
    text: str
    blocks: list[TextBlock] = field(default_factory=list)


@dataclass
class ParsedDocument:
    pages: list[PageContent]
    full_text: str
    raw_metadata: dict


class PDFParser:
    def parse(self, pdf_path: Path) -> ParsedDocument:
        doc = fitz.open(str(pdf_path))
        pages: list[PageContent] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            raw_blocks = page.get_text("blocks")

            blocks = [
                TextBlock(
                    x0=b[0], y0=b[1], x1=b[2], y1=b[3],
                    text=b[4].strip(),
                    block_type=b[6],
                )
                for b in raw_blocks
                if b[4].strip()  # skip empty blocks
            ]
            pages.append(PageContent(page_num=page_num, text=text, blocks=blocks))

        full_text = "\n".join(p.text for p in pages)
        raw_metadata = dict(doc.metadata)
        doc.close()

        return ParsedDocument(pages=pages, full_text=full_text, raw_metadata=raw_metadata)

    def iter_pages(self, pdf_path: Path) -> Iterator[PageContent]:
        doc = fitz.open(str(pdf_path))
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                yield PageContent(
                    page_num=page_num,
                    text=page.get_text("text"),
                )
        finally:
            doc.close()

    def extract_pdf_metadata(self, pdf_path: Path) -> dict:
        doc = fitz.open(str(pdf_path))
        meta = dict(doc.metadata)
        doc.close()
        return meta
