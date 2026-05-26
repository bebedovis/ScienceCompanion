from abc import ABC, abstractmethod
import asyncio 

class Embedder(ABC):
    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        pass

    async def embed_query(self, query: str) -> list[float]:
        embeddings = await self.embed_texts([query])
        return embeddings[0]

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass # embedding vector dimension

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass # model identifier

class HuggingFaceEmbedder(Embedder):
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: str = "cuda") -> None:
        from sentence_transformers import SentenceTransformer
        import torch

        self._model_name = model_name
        self._device = "cuda" if torch.cuda.is_available() and device != "cpu" else "cpu"
        self._model = SentenceTransformer(model_name, device=self._device)
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
            ).tolist(),
        )
        return embeddings
    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension() or 1024
    @property
    def model_name(self) -> str:
        return self._model_name

class OpenAIEmbedder(Embedder):
    def __init__(self, api_key: str, model: str = "text-embedding-3-large") -> None:
        from openai import AsyncOpenAI
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key)

        match self._model:
            case "text-embedding-3-large":
                self._dimension = 3072
            case "text-embedding-3-small":
                self._dimension = 1536
            case "text-embedding-ada-002":
                self._dimension = 1536
            case _:
                self._dimension = 1536

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        cleaned = [t.replace("\n", " ") for t in texts]
        response = await self._client.embeddings.create(
            input=cleaned,
            model=self._model,
        )
        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        return self._dimension
    @property 
    def model_name(self) -> str:
        return self._model

