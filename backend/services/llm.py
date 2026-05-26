from abc import ABC, abstractmethod
from typing import AsyncIterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage


class LLM(ABC):
    @abstractmethod
    def get_chat_model(self) -> BaseChatModel:
        pass # Return a LangChain-compatible chat model for use in LangGraph nodes.

    @abstractmethod
    async def health_check(self) -> bool:
        pass # Return True if the provider is reachable.

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

class Ollama(LLM):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        temperature: float = 0.1,
    ) -> None:
        from langchain_ollama import ChatOllama

        self._base_url = base_url
        self._model = model
        self._temperature = temperature
        self._chat_model = ChatOllama(
            model=self._model,
            base_url=self._base_url,
            temperature=self._temperature,
        )

    def get_chat_model(self) -> BaseChatModel:
        return self._chat_model

    async def health_check(self) -> bool:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self._base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

class OpenAI(LLM):
    def __init__(
        self,
        api_key,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.1,
    ) -> None:
        from langchain_openai import ChatOpenAI

        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._temperature = temperature
        self._chat_model = ChatOpenAI(
            model=self._model,
            api_key=self._api_key,
            base_url=self._base_url,
            temperature=self._temperature,
            streaming=True,
        )

    def get_chat_model(self) -> BaseChatModel:
        return self._chat_model

    async def health_check(self) -> bool:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    f"{self._base_url}/models",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                return r.status_code == 200
        except Exception:
            return False

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model
