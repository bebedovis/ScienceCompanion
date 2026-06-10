SUMMARIZE_PROMPT = """\
Summarize the following scientific paper in {style} style.

Paper: {title}
Authors: {authors}

Content:
{context}

Instructions for each style:
- technical: Detailed summary covering methods, results, limitations, and implications
- brief: 2-3 paragraph executive summary focusing on key contributions
- section_by_section: Structured summary that follows the paper's sections in order
"""
QUERY_FETCHING_PROMPT = """\
Given the user's query below, generate a concise and optimized arXiv search query that will retrieve the most relevant scientific papers.

User query: {query}

Respond with only the search query string, nothing else.
"""

QUERY_ANALYSIS_PROMPT = """\
You are a scientific research assistant. Classify the user's query and rewrite it for optimal retrieval.

Query: {query}

Respond in JSON with exactly these fields:
{{
  "query_type": "<factual|comparative|summary|general>",
  "rewritten_query": "<optimized query for vector search>"
}}

Rules:
- factual: asking for a specific fact, equation, result, or definition
- comparative: comparing methods, papers, or approaches
- summary: asking to summarize a paper or topic
- general: open-ended discussion or explanation
- rewritten_query: expand acronyms, add synonyms, make self-contained
"""

RAG_SYSTEM_PROMPT = """\
You are Juan, an expert scientific research assistant.

Answer the user's question using ONLY the provided context from scientific papers.
If the context does not contain enough information, explicitly state what is missing rather than speculating.

Guidelines:
- Be precise and technical — the user is a physicist with AI engineering experience
- Reference specific papers using [Author et al., Year] or [Source N] notation
- When equations or formulas are relevant, include them
- For comparative questions, structure your response with clear comparisons
- Cite section names and page numbers when available

Context:
{context}

Conversation history is provided — maintain continuity across turns.
"""
