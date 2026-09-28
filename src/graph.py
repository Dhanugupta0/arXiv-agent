"""
The agentic state graph — orchestrator.

Connects all pipeline nodes (1→7) with conditional routing:

  1. Query Understanding    — regex-based intent detection
  2. arXiv Retrieval        — fetch by ID or search by topic (+ query expansion)
  3. Selection / Ranking    — LLM-as-judge picks best candidate
  4. Fetch & Parse          — download PDF, multi-engine extraction
  5. Chunk & Embed          — split text, embed via Jina AI, store in ChromaDB
  6. Summarize              — structured briefing via Groq
  7. QA (per question)      — grounded RAG answers

The graph is a custom state machine.  Each node reads from / writes to a
shared AgentState object.  run_ingest() threads the state through nodes
1→6, then the CLI drives node 7 in a loop.
"""
from __future__ import annotations

from typing import List

from . import arxiv_api, pdf_parser, vectorstore
from .briefing import generate_briefing
from .chunking import prepare_chunks
from .models import AgentState, PaperMetadata
from .qa import answer_question
from .config import MAX_SEARCH_RESULTS, MAX_EXPANSION_ATTEMPTS, has_groq_key


class ArxivDigestGraph:
    """
    State graph orchestrator.

    Usage:
        graph = ArxivDigestGraph()
        state = graph.run_ingest("KV-cache compression for LLMs")
        # state.briefing is now populated; state.collection_name points to ChromaDB
        answer = graph.ask(state, "What datasets were used?")
    """

    # ── Node 1: Query Understanding ──────────────────────────────────────
    def query_understanding(self, state: AgentState) -> None:
        intent, extracted_id, keywords = arxiv_api.parse_query_intent(state.input_query)
        state.query_intent = intent
        state.extracted_id = extracted_id
        state.search_keywords = keywords

    # ── Node 2: arXiv Retrieval (+ expansion) ────────────────────────────
    def arxiv_retrieval(self, state: AgentState) -> None:
        if state.query_intent == "direct_id":
            paper = arxiv_api.fetch_by_id(state.extracted_id)
            if paper is None:
                state.status = "no_results"
                state.log_error(f"No arXiv paper found for ID '{state.extracted_id}'.")
                return
            state.candidate_papers = [paper]
            return

        # Topic search with query expansion fallback
        results = arxiv_api.search_by_topic(state.search_keywords, max_results=MAX_SEARCH_RESULTS)
        while not results and state.expansion_attempts < MAX_EXPANSION_ATTEMPTS:
            expanded = arxiv_api.expand_query(state.search_keywords)
            state.expansion_attempts += 1
            if expanded == state.search_keywords:
                break
            state.search_keywords = expanded
            results = arxiv_api.search_by_topic(state.search_keywords, max_results=MAX_SEARCH_RESULTS)

        if not results:
            state.status = "no_results"
            state.log_error(
                f"No arXiv results for topic '{state.search_keywords}' even after query expansion."
            )
            return
        state.candidate_papers = results

    # ── Node 3: Selection / Ranking ──────────────────────────────────────
    def selection_ranking(self, state: AgentState) -> None:
        candidates: List[PaperMetadata] = state.candidate_papers
        if len(candidates) == 1:
            state.selected_paper = candidates[0]
            return

        # Use LLM-as-judge (Groq or Mock fallback) to pick the most relevant paper
        chosen = self._llm_rank(state.search_keywords or state.input_query, candidates)
        if chosen is not None:
            state.selected_paper = chosen
            return

        # Fallback: pick the first (most recent) result
        state.selected_paper = candidates[0]

    def _llm_rank(self, query: str, candidates: List[PaperMetadata]):
        """LLM-as-judge: pick the most relevant paper from candidates."""
        from . import llm  # deferred import
        listing = "\n".join(
            f"{i+1}. [{c.arxiv_id}] {c.title}\n   Abstract: {c.abstract[:400]}"
            for i, c in enumerate(candidates)
        )
        system = (
            "You are selecting the single most relevant paper for a user's query. "
            "Respond with ONLY the number of the best match, nothing else."
        )
        user = f"Query: {query}\n\nCandidates:\n{listing}\n\nWhich number is the most relevant?"
        try:
            provider = llm.get_llm("auto")
            raw = provider.complete(system, user).strip()
            digits = "".join(ch for ch in raw if ch.isdigit())
            if digits:
                n = int(digits)
                if 1 <= n <= len(candidates):
                    return candidates[n - 1]
        except Exception:
            pass
        return None

    # ── Node 4: Fetch & Parse ────────────────────────────────────────────
    def fetch_and_parse(self, state: AgentState) -> None:
        paper = state.selected_paper
        pdf_bytes = pdf_parser.download_pdf(paper.pdf_url)

        if pdf_bytes is None:
            state.log_error("PDF download failed (network error or unreachable URL).")
            state.raw_text = paper.abstract
            state.parse_method = "abstract_only"
            state.parse_healthy = False
            return

        text, method, healthy = pdf_parser.parse_pdf(pdf_bytes)
        if not healthy:
            state.log_error(f"PDF parsing produced unhealthy text (method: {method}).")
            state.raw_text = paper.abstract
            state.parse_method = "abstract_only"
            state.parse_healthy = False
            return

        state.raw_text = text
        state.parse_method = method
        state.parse_healthy = True

    # ── Node 5: Chunk & Embed ────────────────────────────────────────────
    def chunk_and_embed(self, state: AgentState) -> None:
        chunks = prepare_chunks(state.raw_text)
        if not chunks:
            chunks = prepare_chunks(state.selected_paper.abstract)
        state.chunks = chunks

        # Embed via sentence-transformers (local) and store in ChromaDB
        state.collection_name = vectorstore.index_chunks(
            chunks, state.selected_paper.arxiv_id
        )

    # ── Node 6: Summarize ────────────────────────────────────────────────
    def summarize(self, state: AgentState) -> None:
        degraded = state.parse_method in ("abstract_only", "failed")
        state.briefing = generate_briefing(
            state.selected_paper,
            state.raw_text or state.selected_paper.abstract,
            degraded,
        )
        state.status = "awaiting_qa"

    # ── Node 7: QA (per question) ────────────────────────────────────────
    def ask(self, state: AgentState, question: str) -> str:
        state.messages.append({"role": "user", "content": question})
        answer = answer_question(state, question)
        state.messages.append({"role": "assistant", "content": answer})
        return answer

    # ── Full ingest pipeline: nodes 1 → 6 ────────────────────────────────
    def run_ingest(self, input_query: str) -> AgentState:
        state = AgentState(input_query=input_query)

        self.query_understanding(state)
        self.arxiv_retrieval(state)
        if state.status == "no_results":
            return state

        self.selection_ranking(state)
        self.fetch_and_parse(state)
        self.chunk_and_embed(state)
        self.summarize(state)

        return state
