import asyncio
import urllib.request
from pathlib import Path

import arxiv

try:
    from backend.data_types import Document, DocumentMetadata
except: 
    import sys 
    import os 
    sys.path.append(os.path.abspath(".."))
    from backend.data_types import Document,DocumentMetadata


class ArxivService:
    def __init__(self):
        self._client = arxiv.Client()

    async def search(self, query: str, max_results: int = 5) -> list[arxiv.Result]:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: list(self._client.results(arxiv.Search(query=query, max_results=max_results))),
        )
        return results

    async def download(self, result: arxiv.Result, dirpath: Path) -> Path:
        filename = result.get_short_id().replace("/", "_") + ".pdf"
        pdf_path = dirpath / filename
        pdf_url = result.pdf_url
        if not pdf_url:
            raise ValueError(f"No PDF URL for {result.get_short_id()}")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: urllib.request.urlretrieve(pdf_url, str(pdf_path)))
        return pdf_path

    def result_to_document(self, result: arxiv.Result, pdf_path: Path) -> Document:
        return Document(
            id=result.get_short_id(),
            filename=pdf_path.name,
            file_path=str(pdf_path),
            metadata=DocumentMetadata(
                title=result.title,
                authors=[a.name for a in result.authors],
                year=result.published.year,
                doi=result.doi,
                abstract=result.summary,
                subject_tags=list(result.categories),
                journal=result.journal_ref,
            ),
        )
async def main():

    import os 
    arxiv = ArxivService()
    results = await arxiv.search("optical tweezers")
    for result in results: 
        print(result.title)
        print([a.name for a in result.authors])



if __name__ =="__main__": 
    asyncio.run(main())


