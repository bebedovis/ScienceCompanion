from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from backend.rag.state import RAGState
from backend.rag.nodes import (
    arxiv_query_node,
    fetch_papers,
    query_analysis_node,
    retrieval_node,
    reranking_node,
    generation_node,
    citation_node,
)


def build_rag_graph(checkpointer=None):
    """
    Build and compile the RAG state graph.

    Node flow:
      arxiv_query → fetch_papers → query_analysis → retrieval → reranking → generation → citation → END

    Services are passed via RunnableConfig["configurable"] at invocation time.
    """
    graph = StateGraph(RAGState)

    graph.add_node("arxiv_query", arxiv_query_node)
    graph.add_node("fetch_papers", fetch_papers)
    graph.add_node("query_analysis", query_analysis_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("reranking", reranking_node)
    graph.add_node("generation", generation_node)
    graph.add_node("citation", citation_node)

    graph.add_edge(START, "arxiv_query")
    graph.add_edge("arxiv_query", "fetch_papers")
    graph.add_edge("fetch_papers", "query_analysis")
    graph.add_edge("query_analysis", "retrieval")
    graph.add_edge("retrieval", "reranking")
    graph.add_edge("reranking", "generation")
    graph.add_edge("generation", "citation")
    graph.add_edge("citation", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)
