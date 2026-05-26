import re
import httpx
from typing import Optional

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
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(
                    f"https://export.arxiv.org/abs/{aid}",
                    headers={"User-Agent": "ScienceCompadre/0.1"},
                )
                r.raise_for_status()
        except Exception:
            return None

        html = r.text
        meta = DocumentMetadata(arxiv_id=arxiv_id)

        title_m = re.search(r'<h1 class="title mathjax">.*?<span[^>]*>(.*?)</span>', html, re.DOTALL)
        if title_m:
            meta.title = re.sub(r"<[^>]+>", "", title_m.group(1)).strip()

        authors_m = re.findall(r'class="author"[^>]*>.*?<a[^>]*>(.*?)</a>', html, re.DOTALL)
        meta.authors = [re.sub(r"<[^>]+>", "", a).strip() for a in authors_m]

        abstract_m = re.search(
            r'<blockquote class="abstract mathjax">.*?<span[^>]*>(.*?)</span>',
            html, re.DOTALL
        )
        if abstract_m:
            meta.abstract = re.sub(r"<[^>]+>", "", abstract_m.group(1)).strip()

        year_m = _YEAR_RE.search(html[:3000])
        if year_m:
            meta.year = int(year_m.group())

        return meta

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
