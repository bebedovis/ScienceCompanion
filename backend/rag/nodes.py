import json

from annotated_doc import Doc
import tiktoken
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from backend.rag.prompts import QUERY_ANALYSIS_PROMPT, QUERY_FETCHING_PROMPT, RAG_SYSTEM_PROMPT
from backend.rag.state import RAGState

_MAX_CONTEXT_TOKENS = 6000
_enc = tiktoken.get_encoding("cl100k_base")

def APAstyle_citation(meta:dict, source_num = 1):
    authors = meta.get("authors",[])
    # why joining in datatypes and splitting here, because chroma doesnt allow me to stroe lists jeje 
    if isinstance(authors, str): 
        authors = [a.strip() for a in authors.split(",")] if authors else []

    if len(authors) ==0: 
        author_name = "n.a."
    elif len(authors)==1:
        author_name = authors[0]
    else: 
        author_name = authors[0] +", et al. "
    year = meta.get("year", 0)
    if year ==0 : 
        year = "n.d."
    apa_citation = ""
    apa_citation += f"[Source {source_num}]:  "+ author_name
    apa_citation +=f"({year}) {meta.get('title', 'Unknown')} "

    apa_citation +=f"{meta.get('journal', '')}, {meta.get('section', '')}, "
    apa_citation +=f"p.{meta.get('page', '')}\n"
    return apa_citation


def build_context(chunks: list[dict]):
    full_context = ""
    used_tokens = 0
    chunks_included = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        context = f"[Context {i+1}]: {chunk['text']}\n"
        context += APAstyle_citation(meta, i+1)
        token_count = len(_enc.encode(context))
        if used_tokens + token_count > _MAX_CONTEXT_TOKENS:
            break
        full_context += context
        used_tokens += token_count
        chunks_included.append(chunk)
    return full_context, chunks_included


async def arxiv_query_node(state:RAGState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"].get_chat_model()
    prompt = QUERY_FETCHING_PROMPT.format(query=state["query"])
    response = await llm.ainvoke([
        SystemMessage(content="You are a helpful assistant that generates optimized search queries for scientific papers."),
        HumanMessage(content=prompt),
    ])
    optimized_query = response.content.strip()
    return {"arxiv_query": optimized_query}

async def fetch_papers(state: RAGState, config: RunnableConfig) -> dict:
    arxiv_service = config["configurable"]["arxiv_service"]
    ingest_fn = config["configurable"]["ingest_fn"]
    document_registry = config["configurable"]["document_registry"]
    download_dir = config["configurable"]["download_dir"]

    query = state.get("arxiv_query", "")
    if not query:
        return {"arxiv_results": []}

    already_indexed = {doc.filename for doc in document_registry.list_all()}
    results = await arxiv_service.search(query)

    fetched_docs = []
    for result in results:
        filename = result.get_short_id().replace("/", "_") + ".pdf"
        if filename in already_indexed:
            print(f"  [arxiv] already indexed: {result.title[:60]}")
            continue
        print(f"  [arxiv] downloading: {result.title[:60]}")
        try:
            pdf_path = await arxiv_service.download(result, download_dir)
            await ingest_fn(pdf_path)
            print(pdf_path)
            fetched_docs.append(arxiv_service.result_to_document(result, pdf_path))
        except Exception as exc:
            print(f"  [arxiv] failed {result.get_short_id()}: {exc}")

    return {"arxiv_results": fetched_docs}
    

async def query_analysis_node(state: RAGState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"].get_chat_model()
    prompt = QUERY_ANALYSIS_PROMPT.format(query=state["query"])
    response = await llm.ainvoke([
        SystemMessage(content="You are a query classifier. Respond only with valid JSON."),
        HumanMessage(content=prompt),
    ])
    try:
        parsed = json.loads(response.content)
        query_type = parsed.get("query_type", "general")
        rewritten = parsed.get("rewritten_query", state["query"])
    except (json.JSONDecodeError, AttributeError):
        print(f"Warning: could not parse query analysis response: {str(response.content)[:200]}")
        query_type = "general"
        rewritten = state["query"]
    return {"query_type": query_type, "rewritten_query": rewritten}


async def retrieval_node(state: RAGState, config: RunnableConfig) -> dict:
    retriever = config["configurable"]["retriever"]
    top_k = int(config["configurable"].get("top_k", 20))
    query = state.get("rewritten_query") or state["query"]
    doc_filter = state.get("doc_filter")
    chunks = await retriever(query, n_results=top_k, doc_filter=doc_filter)
    return {"retrieved_chunks": chunks}


async def reranking_node(state: RAGState, config: RunnableConfig) -> dict:
    reranker = config["configurable"]["reranker"]
    top_n = int(config["configurable"].get("rerank_top_n", 6))
    query = state.get("rewritten_query") or state["query"]
    chunks = state["retrieved_chunks"]
    reranked_chunks = await reranker.rerank(query, chunks, top_n)
    return {"reranked_chunks": reranked_chunks}


async def generation_node(state: RAGState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"].get_chat_model()
    chunks = state.get("reranked_chunks")
    if not chunks:
        no_context_msg = (
            "I couldn't find relevant information in the uploaded papers to answer this question. "
            "Please upload the relevant papers or rephrase your query."
        )
        return {"response": no_context_msg, "context": "", "messages": [AIMessage(content=no_context_msg)]}

    context, chunks_included = build_context(chunks)
    system_prompt = RAG_SYSTEM_PROMPT.format(context=context)
    history = state.get("messages", [])
    messages = [SystemMessage(content=system_prompt), *history, HumanMessage(content=state["query"])]
    response = await llm.ainvoke(messages, config=config)
    response_text = response.content
    return {
        "response": response_text,
        "context": context,
        "reranked_chunks": chunks_included,
        "messages": [HumanMessage(content=state["query"]), AIMessage(content=response_text)],
    }


async def citation_node(state: RAGState, config: RunnableConfig) -> dict:
    chunks = state.get("reranked_chunks", [])
    response = state.get("response", "")
    citations: list[str] = []
    seen_papers: set[str] = set()

    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        paper_id = meta.get("paper_id", "")
        source_label = f"[Source {i+1}]"
        if source_label not in response and paper_id in seen_papers:
            continue
        seen_papers.add(paper_id)
        citations.append(APAstyle_citation(meta, i+1))

    return {"citations": citations}
