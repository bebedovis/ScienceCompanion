from backend.config import get_settings, Settings


class ServiceContainer:
    def __init__(self):
        self._embedder = None
        self._semantic_ret = None
        self._keyword_ret = None
        self._hybrid_retriever = None
        self._reranker = None
        self._llm = None
        self._rag_graph = None
        self._document_registry = None
        self._api_service = None

    @property
    def arxiv_service(self):
        if self._api_service is None:
            raise RuntimeError("ArXiv service not initialized")
        return self._api_service

    def set_arxiv_service(self):
        if self._api_service is not None:
            raise RuntimeError("ArXiv service already initialized")
        from backend.services.API import ArxivService
        self._api_service = ArxivService()
    @property
    def embedder(self):
        if self._embedder is None:
            raise RuntimeError("Embedding service not initialized")
        return self._embedder

    @embedder.setter
    def embedder(self, s: Settings):
        if self._embedder is not None:
            raise RuntimeError("Embedding service already initialized")
        from backend.services.embeddings import HuggingFaceEmbedder, OpenAIEmbedder
        if s.embedding_provider == "huggingface":
            self._embedder = HuggingFaceEmbedder(model_name=s.embedding_model, device=s.embedding_device)
        else:
            self._embedder = OpenAIEmbedder(api_key=s.openai_api_key, model=s.embedding_model)

    @property
    def semantic_ret(self):
        if self._semantic_ret is None:
            raise RuntimeError("Semantic retriever not initialized")
        return self._semantic_ret

    @semantic_ret.setter
    def semantic_ret(self, s: Settings):
        if self._semantic_ret is not None:
            raise RuntimeError("Semantic retriever already initialized")
        if self._embedder is None:
            raise RuntimeError("Embedder must be initialized before semantic retriever")
        from backend.services.retrievers import SemanticRetriever
        self._semantic_ret = SemanticRetriever(
            persist_dir=s.chroma_persist_dir,
            collection_name=s.chroma_collection_name,
            embedder=self.embedder,
        )

    @property
    def keyword_ret(self):
        if self._keyword_ret is None:
            raise RuntimeError("Keyword retriever not initialized")
        return self._keyword_ret

    def set_keyword_ret(self):
        if self._keyword_ret is not None:
            raise RuntimeError("Keyword retriever already initialized")
        from backend.services.retrievers import KeywordRetriever
        self._keyword_ret = KeywordRetriever()

    @property
    def hybrid_retriever(self):
        if self._hybrid_retriever is None:
            raise RuntimeError("Hybrid retriever not initialized")
        return self._hybrid_retriever

    def set_hybrid_retriever(self):
        if self._hybrid_retriever is not None:
            raise RuntimeError("Hybrid retriever already initialized")
        from backend.services.retrievers import HybridRetriever
        self._hybrid_retriever = HybridRetriever(
            keyword_retriever=self.keyword_ret,
            semantic_retriever=self.semantic_ret,
        )

    @property
    def reranker(self):
        if self._reranker is None:
            raise RuntimeError("Reranker not initialized")
        return self._reranker

    def set_reranker(self):
        if self._reranker is not None:
            raise RuntimeError("Reranker already initialized")
        from backend.services.reranker import ReRanker
        self._reranker = ReRanker()

    @property
    def llm(self):
        if self._llm is None:
            raise RuntimeError("LLM not initialized")
        return self._llm

    @llm.setter
    def llm(self, s: Settings):
        if self._llm is not None:
            raise RuntimeError("LLM already initialized")
        from backend.services.llm import Ollama, OpenAI
        if s.llm_provider == "ollama":
            self._llm = Ollama(base_url=s.ollama_base_url, model=s.ollama_model)
        else:
            self._llm = OpenAI(api_key=s.openai_api_key, model=s.openai_model, base_url=s.openai_base_url)

    @property
    def rag_graph(self):
        if self._rag_graph is None:
            raise RuntimeError("RAG graph not initialized")
        return self._rag_graph

    def set_rag_graph(self):
        if self._rag_graph is not None:
            raise RuntimeError("RAG graph already initialized")
        from backend.rag.graph import build_rag_graph
        self._rag_graph = build_rag_graph()

    @property
    def document_registry(self):
        if self._document_registry is None:
            raise RuntimeError("Document registry not initialized")
        return self._document_registry

    @document_registry.setter
    def document_registry(self, s: Settings):
        if self._document_registry is not None:
            raise RuntimeError("Document registry already initialized")
        from backend.chroma_client import DocumentRegistry
        self._document_registry = DocumentRegistry(s.upload_path)

    def _init_keyword_retriever(self):
        existing_docs = self.document_registry.list_all()
        if self._semantic_ret is None:
            raise RuntimeError("Semantic retriever must be set before initializing keyword retriever")
        if self._keyword_ret is None:
            raise RuntimeError("Keyword retriever must be set before initializing keyword retriever")
        if existing_docs:
            all_chunks = []
            for doc in existing_docs:
                all_chunks.extend(self._semantic_ret.get_all_chunks_from_paper(doc.id))
            self._keyword_ret.build(all_chunks)

    async def startup(self) -> None:
        s = get_settings()
        self.embedder = s
        self.semantic_ret = s
        self.set_keyword_ret()
        self.set_hybrid_retriever()
        self.set_reranker()
        self.llm = s
        self.document_registry = s
        self.set_rag_graph()
        self._init_keyword_retriever()
        self.set_arxiv_service()
