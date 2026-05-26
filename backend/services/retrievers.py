import re
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings as ChromaSettings
from backend.data_types import Chunk, Document
from backend.services.embeddings import Embedder


class KeywordRetriever:
    """BM25 keyword search over indexed chunks."""

    def __init__(self):
        self._index = None
        self._chunk_ids = []
        self._chunk_metadata = []
        self._chunk_texts = []

    @staticmethod
    def split(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def build(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        self._chunk_ids = [c["id"] for c in chunks]
        self._chunk_texts = [c["text"] for c in chunks]
        self._chunk_metadata = [c["metadata"] for c in chunks]
        corpus = [self.split(c["text"]) for c in chunks]
        self._index = BM25Okapi(corpus)

    def query(self, query: str, n_results: int = 20) -> list[dict]:
        if self._index is None:
            return []
        query_tokens = self.split(query)
        scores = self._index.get_scores(query_tokens)
        rankings = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        chunks = []
        for idx, score in rankings[:n_results]:
            if score == 0.0:
                break
            chunks.append({
                "id": self._chunk_ids[idx],
                "text": self._chunk_texts[idx],
                "metadata": self._chunk_metadata[idx],
                "bm25_score": float(score),
                "score": float(score),
            })
        return chunks


class SemanticRetriever:
    """Vector similarity search using ChromaDB."""

    def __init__(self, persist_dir: str, collection_name: str, embedder: Embedder) -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = embedder

    async def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]], doc: Document) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.to_chroma_metadata(doc) for c in chunks],
        )

    async def query(self, query: str, n_results: int = 20, where=None) -> list[dict]:
        kwargs = {}
        if where:
            kwargs["where"] = where
        query_embedding = await self._embedder.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
            **kwargs,
        )
        if results is None:
            return []
        chunks = []
        for i in range(len(results["ids"][0])):
            dist = results["distances"][0][i]
            chunks.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": dist,
                "score": 1.0 - dist,
            })
        return chunks

    async def delete_paper(self, paper_id: str) -> int:
        results = self._collection.get(where={"paper_id": paper_id})
        ids = results["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def get_all_chunks_from_paper(self, paper_id: str) -> list[dict]:
        results = self._collection.get(
            where={"paper_id": paper_id},
            include=["documents", "metadatas"],
        )
        return [
            {"id": rid, "text": rtext, "metadata": rmeta}
            for rid, rtext, rmeta in zip(results["ids"], results["documents"], results["metadatas"])
        ]

    @property
    def chunk_count(self) -> int:
        return self._collection.count()


class HybridRetriever:
    """Combines BM25 keyword search and semantic vector search."""

    def __init__(self, keyword_retriever: KeywordRetriever, semantic_retriever: SemanticRetriever) -> None:
        self._keyword = keyword_retriever
        self._semantic = semantic_retriever

    async def __call__(self, query: str, n_results: int = 20, doc_filter: list[str] | None = None) -> list[dict]:
        where = {"paper_id": {"$in": doc_filter}} if doc_filter else None
        semantic_results = await self._semantic.query(query, n_results=n_results, where=where)
        keyword_results = self._keyword.query(query, n_results=n_results)

        if doc_filter:
            keyword_results = [r for r in keyword_results if r["metadata"].get("paper_id") in doc_filter]

        if not keyword_results:
            return semantic_results

        return self.merge(semantic_results, keyword_results)

    @staticmethod
    def merge(sem_ranking, key_ranking) -> list[dict]:
        scores, chunk_dic = {}, {}
        for ranking in [sem_ranking, key_ranking]:
            for position, chunk in enumerate(ranking):
                cid = chunk["id"]
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + position + 1)
                chunk_dic[cid] = chunk
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [{**chunk_dic[cid], "rrf_score": scores[cid]} for cid in sorted_ids]
