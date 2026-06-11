import asyncio
import re
from typing import Optional

import arxiv

from backend.data_types import DocumentMetadata
from backend.services.ingestion.pdf_parser import ParsedDocument


_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


class MetadataExtractor:
    async def extract(self, parsed: ParsedDocument, filename: str) -> DocumentMetadata:
        meta = DocumentMetadata()

        # Try PDF metadata first
        raw = parsed.raw_metadata
        if raw.get("title"):
            meta.title = raw["title"].strip()
        if raw.get("author"):
            meta.authors = [a.strip() for a in raw["author"].split(";") if a.strip()]

        # Detect arXiv ID from text
        arxiv_id = self._find_arxiv_id(parsed.full_text, filename)
        if arxiv_id:
            meta.arxiv_id = arxiv_id
            enriched = await self._fetch_arxiv_metadata(arxiv_id)
            if enriched:
                meta = enriched

        # Fallback: infer title from first non-empty line
        if not meta.title:
            meta.title = self._infer_title(parsed.full_text)

        # Fallback: infer year from text
        if not meta.year:
            meta.year = self._infer_year(parsed.full_text)

        # Abstract from text if not fetched
        if not meta.abstract:
            meta.abstract = self._extract_abstract(parsed.full_text)

        return meta

    def _find_arxiv_id(self, text: str, filename: str) -> Optional[str]:
        for source in (filename, text[:2000]):
            m = _ARXIV_ID_RE.search(source)
            if m:
                return m.group(1)
        return None

    async def _fetch_arxiv_metadata(self, arxiv_id: str) -> Optional[DocumentMetadata]:
        clean_id = _ARXIV_ID_RE.search(arxiv_id)
        if not clean_id:
            return None
        aid = clean_id.group(1).split("v")[0]

        try:
            client = arxiv.Client()
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: list(client.results(arxiv.Search(id_list=[aid], max_results=1))),
            )
            if not results:
                return None
            result = results[0]
        except Exception:
            return None

        return DocumentMetadata(
            arxiv_id=arxiv_id,
            title=result.title,
            authors=[a.name for a in result.authors],
            abstract=result.summary,
            year=result.published.year if result.published else None,
            doi=result.doi,
            journal=result.journal_ref,
        )

    def _infer_title(self, text: str) -> str:
        for line in text.splitlines():
            line = line.strip()
            if 10 < len(line) < 200 and not line.endswith("."):
                return line
        return ""

    def _infer_year(self, text: str) -> Optional[int]:
        years = [int(y) for y in _YEAR_RE.findall(text[:3000]) if 1990 <= int(y) <= 2030]
        return max(years) if years else None

    def _extract_abstract(self, text: str) -> str:
        m = re.search(
            r"abstract[\s\n]+(.*?)(?=\n\s*\n|\d+\s+introduction)",
            text[:5000], re.IGNORECASE | re.DOTALL,
        )
        if m:
            return m.group(1).strip()[:1500]
        return ""
