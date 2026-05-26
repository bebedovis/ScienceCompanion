import asyncio
from functools import cached_property
from sentence_transformers.cross_encoder import CrossEncoder


class ReRanker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name

    @cached_property
    def model(self) -> CrossEncoder:
        return CrossEncoder(self._model_name)

    async def rerank(self,query: str, chunks: list[dict], top_n: int=6) -> list[dict]:
        if not chunks:
            return []
        pairs = [(query, chunk["text"]) for chunk in chunks]
        loop = asyncio.get_event_loop()

        scores = await loop.run_in_executor(
            None,
            lambda: self.model.predict(pairs).tolist(),
        )

        ranked = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )
        return [{**chunk, "rerank_score": score} for score, chunk in ranked[:top_n]]


