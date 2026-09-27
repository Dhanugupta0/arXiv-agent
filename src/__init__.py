"""
arXiv Digest & QA Agent — Core Package

Modules:
    config      — centralised configuration (env vars, model names, paths)
    models      — data schemas (PaperMetadata, Chunk, Briefing, AgentState)
    arxiv_api   — Node 1 (query understanding) + Node 2 (arXiv retrieval)
    pdf_parser  — Node 4 (PDF download + multi-engine text extraction)
    chunking    — Node 5a (reference stripping + word-based chunking)
    embeddings  — Node 5b (Jina AI embeddings via HTTP API)
    vectorstore — ChromaDB-backed vector store for RAG retrieval
    llm         — Groq API LLM provider
    briefing    — Node 6 (structured executive briefing generation)
    qa          — Node 7 (grounded QA with anti-hallucination prompt)
    graph       — Orchestrator (state graph: nodes 1→7 with conditional edges)
    session     — Session persistence (list / resume previous papers)
    cli         — Rich-powered interactive CLI
"""
