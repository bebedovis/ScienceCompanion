"""
ScienceCompadre — terminal interface.
Usage: python main.py [--papers-dir ./papers]
"""
import asyncio
import shutil
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage

from backend.config import get_settings
from backend.data_types import Document, DocumentStatus
from backend.rag.state import RAGState
from backend.services.container import ServiceContainer
from backend.services.ingestion.chunker import SemanticChunker
from backend.services.ingestion.metadata_extractor import MetadataExtractor
from backend.services.ingestion.pdf_parser import PDFParser
from backend.services.ingestion.section_extractor import SectionExtractor


async def ingest_pdf(container: ServiceContainer, settings, pdf_path: Path) -> None:
    """Copy a PDF into the upload dir, parse it, embed it, and add it to the index."""
    parser = PDFParser()
    section_extractor = SectionExtractor()
    meta_extractor = MetadataExtractor()
    chunker = SemanticChunker(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)

    doc_id = str(uuid.uuid4())
    dest = settings.upload_path / f"{doc_id}.pdf"
    shutil.copy2(pdf_path, dest)

    doc = Document()
    doc.id = doc_id
    doc.filename = pdf_path.name
    doc.file_path = str(dest)

    container.document_registry.insert(doc)
    container.document_registry.update_status(doc.id, DocumentStatus.PROCESSING)

    try:
        parsed = parser.parse(dest)
        metadata = await meta_extractor.extract(parsed, pdf_path.name)
        doc.metadata = metadata
        container.document_registry.insert(doc)

        sections = section_extractor.extract(parsed)
        chunks = chunker.chunk_sections(sections, paper_id=doc.id)
        if not chunks:
            raise ValueError("No text extracted from PDF")

        embeddings = await container.embedder.embed_texts([c.text for c in chunks])
        await container.semantic_ret.add_chunks(chunks, embeddings, doc)

        all_chunks = container.semantic_ret.get_all_chunks_from_paper(doc.id)
        container.keyword_ret.build(all_chunks)

        container.document_registry.update_status(doc.id, DocumentStatus.READY, chunk_count=len(chunks))
        title = doc.metadata.title or pdf_path.name
        print(f"{title} available ({len(chunks)} chunks)")

    except Exception as exc:
        container.document_registry.update_status(doc.id, DocumentStatus.FAILED)
        print(f"{pdf_path.name} failed to be uploaded : {exc}")


async def chat(container: ServiceContainer, settings, query: str, session_id: str) -> tuple[str, list[dict]]:
    config = {
        "configurable": {
            "thread_id": session_id,
            "llm": container.llm,
            "retriever": container.hybrid_retriever,
            "reranker": container.reranker,
            "top_k": settings.default_top_k,
            "rerank_top_n": settings.rerank_top_n,
            "arxiv_service": container.arxiv_service,
            "ingest_fn": lambda path: ingest_pdf(container, settings, path),
            "document_registry": container.document_registry,
            "download_dir": settings.upload_path,
        }
    }
    state: RAGState = {
        "query": query,
        "arxiv_query": "",
        "session_id": session_id,
        "doc_filter": None,
        "messages": [HumanMessage(content=query)],
        "rewritten_query": "",
        "query_type": "general",
        "arxiv_results": [],
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "context": "",
        "response": "",
        "citations": [],
    }
    result = await container.rag_graph.ainvoke(state, config)
    return result.get("response", ""), result.get("citations", [])


async def main() -> None:
    settings = get_settings()

    papers_dir = Path("./papers")
    papers_dir.mkdir(parents=True, exist_ok=True)

    print("Starting up services...")
    container = ServiceContainer()
    await container.startup()

    # Ingest any PDFs not yet in the registry
    already_indexed = {doc.filename for doc in container.document_registry.list_all()}
    new_pdfs = [p for p in sorted(papers_dir.glob("*.pdf")) if p.name not in already_indexed]

    if new_pdfs:
        print(f"\nIndexing {len(new_pdfs)} new paper(s) from {papers_dir}/")
        for pdf in new_pdfs:
            await ingest_pdf(container, settings, pdf)

    ready = [d for d in container.document_registry.list_all() if d.status == DocumentStatus.READY]
    if not ready:
        print(f"\nNo papers indexed yet. Drop PDFs into {papers_dir}/ and run again.")
        return

    print(f"\n{len(ready)} paper(s) ready:")
    for doc in ready:
        print(f"  • {doc.metadata.title or doc.filename}")

    print("\nType your question, or 'quit' to exit.\n")
    session_id = str(uuid.uuid4())

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        print("Thinking...\n")
        response, citations = await chat(container, settings, query, session_id)

        print(response)

        if citations:
            print(f"\n── Sources {'─' * 50}")
            for i,citation in enumerate(citations):
                print(f"[{i}] "+ citation)

if __name__ == "__main__":
    asyncio.run(main())
