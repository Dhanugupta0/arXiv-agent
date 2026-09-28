# 🔬 Autonomous arXiv Paper Digest & QA Agent

A CLI agent that takes a research topic or arXiv ID/URL, retrieves and parses the paper, builds a semantic RAG index over it, produces a structured executive briefing, and answers follow-up questions grounded in the paper's own text.

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **LLM** | Groq API (`llama-3.3-70b-versatile`) | Fast inference, free tier, high quality |
| **Embeddings** | sentence-transformers (`BAAI/bge-small-en-v1.5`) | Local, no API key, works offline, 384-dim |
| **Vector DB** | ChromaDB (persistent, local) | Embeddings persist to disk — no recomputation |
| **PDF Parsing** | PyMuPDF → pypdf → OCR fallback | Multi-engine extraction with graceful degradation |
| **arXiv Access** | Official arXiv API (Atom feed) | No scraping needed |
| **CLI** | Rich | Colour-coded output, spinners, formatted tables |

## Setup

```bash
# 1. Clone & enter the project
cd arxiv-agent

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Add your Groq API key to .env
#    GROQ_API_KEY  — get free at https://console.groq.com
#    Without it, the agent still works using extractive heuristics.
nano .env
```

> **First run note:** The embedding model (`BAAI/bge-small-en-v1.5`, ~130MB) downloads
> automatically from HuggingFace on first use. After that, it works fully offline.

### Groq Free-Tier Rate Limits

The Groq API key is **optional** — the agent runs end-to-end without it using
extractive heuristics. When you do set `GROQ_API_KEY`, be aware of Groq's
free-tier rate limits:

| Model | RPM | RPD | TPM | TPD |
|-------|-----|-----|-----|-----|
| `llama-3.3-70b-versatile` | 30 | 1,000 | 12,000 | 100,000 |
| `openai/gpt-oss-20b` | 30 | 14,400 | 250,000 | 500,000 |
| `qwen/qwen3.8-27b` | 30 | 14,400 | 6,000 | 500,000 |

> Check [console.groq.com/docs/rate-limits](https://console.groq.com/docs/ra
te-limits) for current limits.
> The agent includes automatic retry with backoff on 429 rate-limit errors.

## Usage

```bash
# Digest a paper by topic search
python main.py digest "recent work on KV-cache compression for LLMs"

# Digest a specific paper by arXiv ID
python main.py digest 2401.12345

# Digest by full URL
python main.py digest https://arxiv.org/abs/2401.12345

# Briefing only, skip QA
python main.py digest 2401.12345 --no-qa

# Resume QA on a previously digested paper
python main.py qa 2401.12345

# List all saved sessions
python main.py sessions

# Demo mode (bundled synthetic paper — works with zero configuration)
python main.py demo
```

## Architecture: State Graph

The agent is implemented as an explicit state graph with 7 nodes over a shared `AgentState` object. Each node reads from and mutates this state; the graph runner threads it through nodes 1→6, then holds it at `"awaiting_qa"` for repeated node-7 calls.

```
                         ┌─────────────────────┐
  input_query ──────────▶│ 1. Query             │
                         │ Understanding        │ (regex: arXiv ID/URL vs. free text)
                         └──────────┬───────────┘
                                    │
                     query_intent ──┼── "direct_id" ──────────────┐
                                    │                             │
                            "topic_search"                        │
                                    ▼                             ▼
                         ┌─────────────────────┐       ┌──────────────────┐
                         │ 2. arXiv Retrieval   │       │ 2. arXiv          │
                         │ (topic search,       │       │ Retrieval         │
                         │  top-k by date)      │       │ (fetch by ID)     │
                         └──────────┬───────────┘       └────────┬─────────┘
                              0 results?                          │
                         ┌──────────▼───────────┐                ▼
                         │ query_expansion       │       ┌─────────────────┐
                         │ (heuristic broadening,│       │ status=no_results│
                         │  retry once)          │       │ (terminal)       │
                         └──────────┬───────────┘       └─────────────────┘
                             still 0?  ──▶ status=no_results (terminal)
                                    │
                              >1 candidate
                                    ▼
                         ┌─────────────────────┐
                         │ 3. Selection/Ranking  │  LLM-judge picks best paper
                         └──────────┬───────────┘  (falls back to most-recent
                                    ▼               if no API key)
                         ┌─────────────────────┐
                         │ 4. Fetch & Parse      │  PyMuPDF → pypdf → OCR
                         │                       │  (degrades to abstract-only)
                         └──────────┬───────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ 5. Chunk & Embed      │  ~800-word chunks
                         │                       │  sentence-transformers → ChromaDB
                         └──────────┬───────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ 6. Summarize          │  Groq LLM → structured JSON
                         └──────────┬───────────┘  (or extractive heuristics
                                    ▼               if no API key)
                         status = "awaiting_qa"  ◀── durable pause point
                                    │
                                    ▼
                         ┌─────────────────────┐
                    ┌───▶│ 7. QA (per question)  │  ChromaDB retrieval → Groq
                    │    └──────────┬───────────┘  (or extractive fallback)
                    └───────────────┘  (loops until user exits)
```

## Repo Layout

```
main.py                    CLI entrypoint (digest / qa / sessions / demo)
src/
  __init__.py              Package docstring
  config.py                Centralised config (env vars, model names, paths)
  models.py                Data schemas: PaperMetadata, Chunk, Briefing, AgentState
  arxiv_api.py             Node 1 (intent) + Node 2 (arXiv retrieval, expansion)
  pdf_parser.py            Node 4 (download, parse w/ fallback chain)
  chunking.py              Node 5a (reference stripping, word-based chunking)
  embeddings.py            Node 5b (local embeddings via sentence-transformers)
  vectorstore.py           ChromaDB-backed persistent vector store
  llm.py                   Groq API provider (with retry + backoff)
  briefing.py              Node 6 (structured briefing, LLM + extractive fallback)
  qa.py                    Node 7 (grounded QA, anti-hallucination prompt)
  graph.py                 Orchestrator: state graph with conditional routing
  session.py               JSON-based session persistence
  cli.py                   Rich-powered interactive CLI
sample_data/
  sample_paper.txt         Synthetic paper for demo mode
sessions/                  Created at runtime; saved sessions per arXiv ID
chroma_store/              Created at runtime; persistent ChromaDB embeddings
.env                       API keys (GROQ_API_KEY — optional)
requirements.txt
```

## Where Are Embeddings Stored?

**ChromaDB** (local, persistent, on disk):

- Embeddings are generated by **sentence-transformers** (BAAI/bge-small-en-v1.5) locally
- Stored in `chroma_store/` directory using ChromaDB's persistent client
- Each paper gets its own ChromaDB collection (keyed by arXiv ID + embedding model tag)
- Embeddings are **never recomputed** for already-indexed papers
- Sessions survive across CLI invocations — resume QA without re-embedding
- Switching embedding models creates new collections automatically (no silent corruption)

## Handling Edge Cases

| Scenario | How it's handled |
|----------|--------------------|
| **Zero arXiv results** | Heuristic query expansion (drop quotes, parentheticals, last keyword), retry once, then clean terminal failure |
| **Many arXiv results** | LLM-as-judge ranks candidates by abstract relevance; falls back to most recent |
| **PDF fails to parse** | PyMuPDF → pypdf → OCR fallback chain; degrades to abstract-only (never crashes) |
| **QA hallucination** | System prompt enforces grounding in retrieved chunks only; explicit refusal if answer isn't in paper |
| **Session persistence** | JSON sessions + ChromaDB embeddings survive across CLI invocations |
| **No API keys set** | Runs end-to-end with extractive briefing + extractive QA (zero-config mode) |
| **Groq rate limit (429)** | Automatic retry with exponential backoff (up to 2 retries) |

## Design Decisions & Tradeoffs

- **Groq API only (no multi-provider):** Simplifies the codebase — one reliable LLM provider instead of a complex fallback chain. Groq's free tier is generous with fast inference.
- **Local embeddings (sentence-transformers):** No API key needed. BAAI/bge-small-en-v1.5 is 33M params, 384-dim, runs on CPU. Downloads once (~130MB), then works fully offline.
- **BGE query instruction prefix:** bge-small-en-v1.5 requires an instruction prefix on query embeddings only (not passages) for optimal retrieval quality.
- **Embedding-model-tagged collections:** Collection names include the embedding model tag (e.g. `paper_2401_12345_bge_small_en_v1_5`) so switching models never silently corrupts existing indices.
- **ChromaDB (not pickle/in-memory):** Embeddings persist to disk. Papers indexed once never need re-embedding. Multiple papers coexist without interference.
- **JSON sessions (not pickle):** Human-readable, safe to inspect, no deserialization vulnerabilities. Embeddings are already in ChromaDB.
- **Custom state machine (not LangGraph):** For a mostly-linear pipeline, explicit routing is easier to audit. Same node/state pattern would port trivially to LangGraph.
- **Word-based chunking (not tokenizer-based):** Avoids tiktoken dependency; ~800 words ≈ ~1000 tokens, close enough for academic prose.
- **Abstract-only degradation over hard failure:** A partial, honestly-labelled answer is more useful than a crash.
- **Single-hop RAG:** One top-k lookup per question. Sufficient for most single-paper factual questions; multi-pass re-ranking would be the natural next step.
- **Retry with backoff on Groq errors:** Since Groq is the only remaining external dependency (embeddings are local), a rate-limit blip shouldn't crash the run.
- **Zero-config demo mode:** `python main.py demo` runs end-to-end with an empty .env file — no API keys needed.

## Known Limitations / Next Steps

- OCR fallback needs system-level Poppler + Tesseract binaries
- No multi-paper comparison mode
- Query expansion is heuristic-only, not LLM-driven
- Single-hop RAG — no verification/re-ranking pass
- No automated test suite beyond manual verification
