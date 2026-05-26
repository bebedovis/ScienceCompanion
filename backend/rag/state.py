from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    # Input
    query: str
    session_id: str
    doc_filter: Optional[list[str]]

    # Conversation history managed by LangGraph
    messages: Annotated[list[BaseMessage], add_messages]

    # Retrieval pipeline state
    rewritten_query: str
    query_type: str              # "factual" | "comparative" | "summary" | "general"
    retrieved_chunks: list[dict]
    reranked_chunks: list[dict]

    # Generation state
    context: str
    response: str
    citations: list[dict]
