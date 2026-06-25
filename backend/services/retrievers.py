import re
from opensearchpy import OpenSearch
from rank_bm25 import BM25Okapi
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
    """Vector similarity search using Amazon OpenSearch Service."""

    _INDEX_SETTINGS = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100,
            }
        }
    }

    def __init__(
        self,
        host: str,
        port: int,
        index_name: str,
        embedding_dim: int,
        embedder: Embedder,
        username: str = "",
        password: str = "",
        use_ssl: bool = False,
    ) -> None:
        self._client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=(username, password) if username else None,
            use_ssl=use_ssl,
            verify_certs=False,
            ssl_show_warn=False,
        )
        self._index = index_name
        self._embedder = embedder
        self._ensure_index(embedding_dim)

    def _ensure_index(self, dim: int) -> None:
        if self._client.indices.exists(index=self._index):
            return
        mapping = {
            **self._INDEX_SETTINGS,
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                        },
                    },
                    "text": {"type": "text"},
                    "paper_id": {"type": "keyword"},
                    "section": {"type": "keyword"},
                    "page": {"type": "integer"},
                    "chunk_type": {"type": "keyword"},
                    "title": {"type": "text"},
                    "authors": {"type": "text"},
                    "year": {"type": "integer"},
                    "journal": {"type": "keyword"},
                }
            },
        }
        self._client.indices.create(index=self._index, body=mapping)

    async def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]], doc: Document) -> None:
        if not chunks:
            return
        actions = []
        for chunk, embedding in zip(chunks, embeddings):
            actions.append({"index": {"_index": self._index, "_id": chunk.id}})
            actions.append({"embedding": embedding, "text": chunk.text, **chunk.to_metadata(doc)})
        self._client.bulk(body=actions)

    async def query(self, query: str, n_results: int = 20, doc_filter: list[str] | None = None) -> list[dict]:
        query_embedding = await self._embedder.embed_query(query)
        knn_clause = {"embedding": {"vector": query_embedding, "k": n_results}}

        if doc_filter:
            body = {
                "size": n_results,
                "query": {
                    "bool": {
                        "must": [{"knn": knn_clause}],
                        "filter": [{"terms": {"paper_id": doc_filter}}],
                    }
                },
            }
        else:
            body = {"size": n_results, "query": {"knn": knn_clause}}

        response = self._client.search(index=self._index, body=body)
        chunks = []
        for hit in response["hits"]["hits"]:
            src = hit["_source"]
            chunks.append({
                "id": hit["_id"],
                "text": src["text"],
                "metadata": {k: v for k, v in src.items() if k not in ("embedding", "text")},
                "score": hit["_score"],
            })
        return chunks

    async def delete_paper(self, paper_id: str) -> int:
        response = self._client.delete_by_query(
            index=self._index,
            body={"query": {"term": {"paper_id": paper_id}}},
        )
        return response["deleted"]

    def get_all_chunks_from_paper(self, paper_id: str) -> list[dict]:
        response = self._client.search(
            index=self._index,
            body={
                "size": 10000,
                "query": {"term": {"paper_id": paper_id}},
                "_source": {"excludes": ["embedding"]},
            },
        )
        return [
            {
                "id": hit["_id"],
                "text": hit["_source"]["text"],
                "metadata": {k: v for k, v in hit["_source"].items() if k != "text"},
            }
            for hit in response["hits"]["hits"]
        ]

    @property
    def chunk_count(self) -> int:
        return self._client.count(index=self._index)["count"]


class HybridRetriever:
    """Combines BM25 keyword search and semantic vector search."""

    def __init__(self, keyword_retriever: KeywordRetriever, semantic_retriever: SemanticRetriever) -> None:
        self._keyword = keyword_retriever
        self._semantic = semantic_retriever

    async def __call__(self, query: str, n_results: int = 20, doc_filter: list[str] | None = None) -> list[dict]:
        semantic_results = await self._semantic.query(query, n_results=n_results, doc_filter=doc_filter)
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
