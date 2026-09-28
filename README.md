# 🔬 Autonomous arXiv Paper Digest & QA Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Vector DB: ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![Embeddings: BGE-small](https://img.shields.io/badge/Embeddings-BAAI%2Fbge--small--en--v1.5-brightgreen.svg)](https://huggingface.co/BAAI/bge-small-en-v1.5)
[![LLM: Groq](https://img.shields.io/badge/LLM-Groq%20Cloud-orange.svg)](https://console.groq.com)
[![CLI: Rich](https://img.shields.io/badge/CLI-Rich-pink.svg)](https://github.com/Textualize/rich)

> **An autonomous pair-researcher CLI agent** that turns dense academic papers on arXiv into structured executive briefings and interactive, grounded RAG Q&A sessions. It features deterministic routing, zero-cost heuristic fallbacks, offline capability, and disk-persisted vector storage.

---



### 📸 Live Ingestion Demos in Action

<table width="100%">
  <tr>
    <th width="50%" align="center"><b>Direct Paper ID Digest (<code>python main.py digest 2401.12345</code>)</b></th>
    <th width="50%" align="center"><b>Topic Search Digest (<code>python main.py digest "KV-cache compression for LLMs"</code>)</b></th>
  </tr>
  <tr>
    <td align="center">
      <img src="img/demo_direct_id.png" alt="Direct arXiv ID Ingestion Demo" width="100%" />
    </td>
    <td align="center">
      <img src="img/demo_topic_search.png" alt="Topic Search Ingestion Demo" width="100%" />
    </td>
  </tr>
</table>

---

## ✨ Key Features

- **Dual-Mode Operation (Cloud or 100% Offline):** Runs with Groq API for lightning-fast LLM inference, or completely zero-key and offline with heuristic rule-based summarization and extractive RAG.
- **Deterministic Intent & Query Routing:** Regex patterns classify queries into direct IDs, arXiv URLs, or free-text research topics without burning expensive LLM tokens.
- **Self-Healing Search:** Automated heuristic query expansion broadens search terms if zero papers are returned initially.
- **LLM-as-a-Judge Candidate Ranking:** Evaluates multiple paper candidates by title and abstract to select the most relevant match for your query.
- **Cascading 4-Tier PDF Ingestion:** PyMuPDF (layout-aware blocks) ➔ pypdf ➔ Tesseract OCR ➔ Abstract-only graceful degradation.
- **Local Dense Embeddings:** Runs `BAAI/bge-small-en-v1.5` locally on CPU with asymmetric search prefixes and cosine normalization.
- **Persistent ChromaDB Vector Store:** Isolated collections tagged by paper ID and embedding model. Never re-indexes a paper twice.
- **Strict Anti-Hallucination QA:** System prompts force the LLM to cite excerpts (e.g. `[Excerpt 1]`) and respond with an exact refusal phrase when information is absent.
- **Durable Session Persistence:** Session states and conversation histories are stored in human-readable JSON files, allowing you to resume Q&A anytime.

---

## 🏗️ High-Level System Architecture

<p align="center">
  <img src="img/HLD.png" alt="Autonomous arXiv Digest Agent High Level Design Architecture" width="100%" />
</p>

### 💡 The Architecture in Simple Words (Plain English)

Imagine having a **tireless, super-fast research partner** sitting inside your terminal. Here is how the agent processes your requests in plain, everyday language:

1. **Instant Query Routing (No Wasted Money):** When you type a query, the agent doesn't waste expensive AI tokens asking *"is this a link or a topic?"*. Instead, a lightning-fast regular expression checks if you provided an arXiv ID (`2401.12345`) or an arXiv URL. If yes, it fetches that exact paper immediately. If it's a general topic like `"KV-cache compression"`, it triggers an academic search.
2. **Self-Healing arXiv Search:** The agent queries arXiv's official search engine. If your search terms were too specific and returned 0 papers, it doesn't give up! It automatically loosens quotes, removes parenthetical filters, and drops trailing keywords to retry the search on its own.
3. **Smart Paper Selection (LLM Judge):** When a topic search returns multiple papers, an AI judge reads their titles and abstracts to pick the single most relevant paper for your request.
4. **Bulletproof 4-Tier PDF Reader:** Academic PDFs are notorious for complex two-column layouts, weird fonts, and scanned figures. The agent uses a safety-ladder approach: it tries **PyMuPDF** (preserving two-column reading order) ➔ then **pypdf** ➔ then **OCR image reading**. If the PDF is completely broken or unreachable, it gracefully falls back to using the abstract. **The agent never crashes.**
5. **Local & Free Vector Memory (ChromaDB + BGE Embeddings):** It strips out references and bibliographies to keep search results clean, divides the paper into 800-word chunks, and converts them into mathematical vectors using a **local, 100% offline model (`BGE-small`) running directly on your CPU**. It stores these vectors into a local **ChromaDB** database on disk. You don't pay a single cent for embeddings, and it never re-processes a paper you've already indexed.
6. **Executive Briefing Card:** It feeds the clean text to **Groq's ultra-fast Qwen model** to produce a structured, high-impact briefing (Summary, Problem Statement, Method, Key Results, and Inferred Limitations). If you don't have an API key, an offline heuristic extractor creates the briefing for you.
7. **Anti-Hallucination Q&A with Receipts:** When you ask questions about the paper, the agent retrieves the top 3 relevant paragraphs from ChromaDB and tells the model: *"Answer using ONLY these excerpts and cite them like [Excerpt 1]. If the answer isn't in the text, say you don't know."* No hallucinations, no guessing.
8. **Durable Session Memory:** Everything is saved to clean, human-readable JSON files in `sessions/`. You can close your laptop, come back days later, and run `python main.py qa <arxiv_id>` to continue asking questions instantly.


## 🔄 Agent Loop & Techniques (Start to End)

The agent processes information across seven discrete, measurable pipeline nodes. 

### Detailed Node Breakdown

<details>
<summary><b>🔍 Node 1: Deterministic Query Understanding</b></summary>

- **Technique:** Regex-based zero-cost intent detection.
- **Pattern Matching:**
  - `_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")`
  - `_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(v\d+)?")`
- **Logic:**
  1. Checks if the query contains an explicit arXiv URL.
  2. Checks for exact ID patterns (e.g. `2401.12345`).
  3. Looks for loose IDs in short queries (`"summarize 2401.12345 for me"`).
  4. Otherwise routes to `"topic_search"` with extracted keywords.
- **Why No LLM?** Saves latency and cost for queries with rigid structural identifiers.
</details>

<details>
<summary><b>📡 Node 2: arXiv Retrieval & Heuristic Query Expansion</b></summary>

- **Technique:** Official arXiv API querying using the Atom feed specification.
- **Keyword Enrichment:** Formats free-text search queries as `all:<keywords>` to match across paper title, abstract, and indexed metadata simultaneously.
- **Query Expansion Heuristic:**
  - If `0` candidates are returned, the query expansion loop activates.
  - Strips double-quoted exact-phrase constraints: `re.sub(r'"[^"]*"', "", q)`
  - Strips parenthetical filters: `re.sub(r"\([^)]*\)", "", q)`
  - Removes the least critical / most specific trailing keyword.
  - Retries the arXiv API search automatically once before declaring failure.
</details>

<details>
<summary><b>🏆 Node 3: Selection / Ranking (LLM-as-a-Judge)</b></summary>

- **Technique:** Few-candidate decision arbitration.
- **Logic:**
  - If `len(candidates) == 1`, bypasses the LLM and selects immediately.
  - If multiple candidates exist, builds an enumeration prompt containing candidate titles and the first 400 characters of each abstract.
  - Instructs the LLM to output **only the integer index** of the most relevant match.
  - Parses digits from the response. If parsing fails or the LLM is offline, falls back safely to Candidate 1 (the most recent paper).
</details>

<details>
<summary><b>📥 Node 4: Multi-Engine PDF Fetch & Cascading Parser</b></summary>

- **Technique:** 4-tier cascading fallback for document extraction.
- **Engines:**
  1. **PyMuPDF (`fitz`):** Extracts structural text blocks sorted by coordinates `(round(y, 1), round(x, 1))` to preserve two-column academic reading order.
  2. **pypdf:** Independent pure-Python stream reader fallback.
  3. **OCR (pdf2image + pytesseract):** Rasterizes PDF pages at 200 DPI and performs optical character recognition (optional system dependencies).
  4. **Abstract-Only Degradation:** If all parsers fail or the PDF cannot be downloaded, sets `raw_text = abstract`, `parse_method = "abstract_only"`, and marks `parse_healthy = False`.
- **Health Checks:** A parsed text is only accepted if:
  - Total length $\ge 1000$ characters.
  - Unicode replacement character ratio (`\ufffd`) $\le 2\%$.
  - Alphanumeric character ratio $\ge 40\%$.
</details>

<details>
<summary><b>🧩 Node 5: Noise Stripping, Chunking & Dense Embeddings</b></summary>

- **Technique:** Reference pruning + sliding-window chunking + asymmetric dense retrieval.
- **Reference Stripping:** Detects `References` or `Bibliography` headers past the 40% mark of the paper and discards everything after to prevent bibliography clutter in the vector store.
- **Sliding Window:** 800-word chunks with 150-word overlaps. Word-based tokenization eliminates tokenizer dependencies while maintaining $\sim 1000$ token segments.
- **Dense Embedding Model:** `BAAI/bge-small-en-v1.5` (33M parameters, 384 dimensions, CPU optimized).
- **Asymmetric Prefixing:** Queries receive the BGE instruction prefix:
  `"Represent this sentence for searching relevant passages: "`
  Document passages are embedded without the prefix. This asymmetric encoding boosts recall significantly.
- **Vector Storage:** ChromaDB collection named `paper_{sanitized_id}_{embedding_tag}` using cosine distance. Skips re-indexing if the collection count matches the chunk count.
</details>

<details>
<summary><b>📝 Node 6: Executive Briefing Generation</b></summary>

- **Technique:** Strict JSON schema extraction with heuristic fallback.
- **Structured Fields:**
  - `summary`: One-paragraph overview in plain English.
  - `problem_statement`: Core gap or challenge addressed.
  - `method`: List of concrete architectural/algorithmic points.
  - `key_results`: Benchmark metrics and empirical claims.
  - `limitations`: Inferred or explicitly stated boundaries (guaranteed non-empty).
  - `followup_questions`: 3 provocative technical questions.
- **Resilience:** If the LLM output fails JSON parsing or the LLM is absent, an extractive regex engine scans headers for `method`, `results`, and `limitations` to construct the briefing without crashing.
</details>

<details>
<summary><b>💬 Node 7: Grounded Q&A with Anti-Hallucination Guardrails</b></summary>

- **Technique:** Single-hop RAG with strict factual grounding.
- **Process:**
  1. Computes query embedding using the BGE search prefix.
  2. Retrieves Top-$k$ (default: 3) chunks from ChromaDB by cosine similarity.
  3. Formats chunks as `[Excerpt N, relevance=X.XX]`.
  4. Prompt restricts the LLM to use **ONLY** the provided excerpts and paper info.
  5. Mandates the exact refusal sentence if answers are not present:
     > *"The provided text does not contain sufficient information to answer this question."*
  6. Mandates inline citations (`[Excerpt 1]`).
  7. If offline or no key, returns the highest-scoring passage directly in an extractive answer card.
</details>

---

## 🛠️ Tech Stack & Technical Rationale

| Layer | Component | Technology | Rationale |
|---|---|---|---|
| **Language** | Runtime | **Python 3.10+** | Modern type hinting, rich ecosystem for ML/data/CLI. |
| **LLM Inference** | Primary Provider | **Groq Cloud API** | Sub-second TTFT (Time-to-First-Token), free developer tier, Qwen 2.5 / Llama 3 models. |
| **Default Model** | Foundation Model | `qwen/qwen3.8-27b` | Exceptional structured JSON following, 131k context window, low hallucination rate. |
| **Offline LLM** | Fallback Provider | **MockProvider (Custom)** | Deterministic regex-based extractor for zero-key, zero-network environments. |
| **Embedding Engine** | Local Model | `BAAI/bge-small-en-v1.5` | SOTA MTEB retrieval performance at 33M parameters; runs fast on CPU with no API key needed. |
| **Vector Database** | Storage & Indexing | **ChromaDB (Persistent)** | Embedded vector store on disk (`chroma_store/`); HNSW cosine space; eliminates cloud vector DB overhead. |
| **PDF Extraction** | Multi-Tier Engine | **PyMuPDF + pypdf + Tesseract** | Fast layout-preserving two-column reading order with multi-tier degradation. |
| **arXiv Protocol** | Data Ingestion | **arxiv (Python SDK)** | Direct client wrapper over arXiv's Atom XML syndication API. |
| **State Persistence** | Session Storage | **JSON on Disk (`sessions/`)** | Human-inspectable, safe against pickle vulnerabilities, stores conversation state across CLI runs. |
| **CLI / UI** | Presentation | **Rich** | Spinners, formatted markdown panels, colorful logs, and clean terminal tables. |

---

## 🔌 Major APIs, Protocols & Schemas

### 1. External APIs Used

```mermaid
graph LR
    subgraph ExternalServices [External Services]
        A[arXiv Atom API]
        B[arXiv PDF Gateway]
        C[Groq Cloud LLM API]
        D[Hugging Face Hub]
    end

    subgraph AgentSystem [arXiv Digest Agent]
        E[arxiv_api.py]
        F[pdf_parser.py]
        G[llm.py]
        H[embeddings.py]
    end

    A <-->|HTTP GET /query Atom XML| E
    B <-->|HTTP GET /pdf stream| F
    C <-->|OpenAI-Compatible Chat Completions| G
    D <-->|One-time 130MB Model Download| H
```

- **arXiv Search API:**
  - Protocol: HTTP GET against `export.arxiv.org/api/query` returning Atom 1.0 XML.
  - Endpoints exercised: `Search(query="all:...", sort_by=Relevance)` and `Search(id_list=[arxiv_id])`.
- **Groq Cloud Chat Completions API:**
  - Protocol: HTTPS REST (OpenAI-compatible) endpoint `https://api.groq.com/openai/v1/chat/completions`.
  - Headers: `Authorization: Bearer <GROQ_API_KEY>`.
  - Parameters: `model="qwen/qwen3.8-27b"`, `temperature=0.2`, `max_tokens=4096`.
  - Rate-limit handling: Exponential backoff on HTTP 429 and transient 5xx errors.
- **Hugging Face Hub Model Repository:**
  - Protocol: HTTPS download on first launch for `BAAI/bge-small-en-v1.5` weights ($\sim 130$ MB), cached locally in `~/.cache/huggingface/hub/`.

### 2. Core Data Schemas

```mermaid
classDiagram
    class PaperMetadata {
        +str arxiv_id
        +str title
        +List~str~ authors
        +str abstract
        +str published
        +str pdf_url
        +str abs_url
        +List~str~ categories
        +to_dict() Dict
    }

    class Chunk {
        +int id
        +str text
        +Optional~str~ section_hint
    }

    class Briefing {
        +str title
        +List~str~ authors
        +str arxiv_id
        +str published
        +str link
        +str summary
        +str problem_statement
        +List~str~ method
        +List~str~ key_results
        +List~str~ limitations
        +List~str~ followup_questions
        +bool degraded
        +to_markdown() str
        +to_dict() Dict
    }

    class AgentState {
        +str input_query
        +Optional~str~ query_intent
        +Optional~str~ extracted_id
        +Optional~str~ search_keywords
        +int expansion_attempts
        +List~PaperMetadata~ candidate_papers
        +Optional~PaperMetadata~ selected_paper
        +Optional~str~ raw_text
        +Optional~str~ parse_method
        +bool parse_healthy
        +List~Chunk~ chunks
        +Optional~str~ collection_name
        +Optional~Briefing~ briefing
        +List~Dict~ messages
        +str status
        +List~str~ errors
        +log_error(str msg)
    }

    AgentState o-- PaperMetadata : selected_paper
    AgentState o-- Chunk : chunks
    AgentState o-- Briefing : briefing
```

---



## 🛡️ Cascading Fallbacks & Resilience Matrix

```mermaid
graph TD
    subgraph Failures [Failure Scenarios]
        F1[arXiv API: Zero search results]
        F2[PDF Gateway: Download failed or HTTP 403]
        F3[PDF Parser: Garbled text / scans]
        F4[Groq API: Missing API Key]
        F5[Groq API: HTTP 429 Rate Limit]
        F6[ChromaDB: Session reload]
    end

    subgraph Mitigations [Agentic Resilience Strategy]
        M1[Query Expansion: Strip quotes & drop last keyword ➔ Retry Search]
        M2[Graceful Degradation: Revert to Abstract-only raw text]
        M3[Parser Fallback: PyMuPDF ➔ pypdf ➔ OCR ➔ Abstract fallback]
        M4[MockProvider: Heuristic regex briefing & direct chunk extractive QA]
        M5[Exponential Backoff: Retry 1-2s ➔ Exhaustion drops to MockProvider]
        M6[Idempotent Read: Reuse existing collection; skip re-embedding]
    end

    F1 ==> M1
    F2 ==> M2
    F3 ==> M3
    F4 ==> M4
    F5 ==> M5
    F6 ==> M6

    classDef fail fill:#ffebee,stroke:#c62828,stroke-width:1px,color:#b71c1c;
    classDef fix fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px,color:#1b5e20;
    class F1,F2,F3,F4,F5,F6 fail;
    class M1,M2,M3,M4,M5,M6 fix;
```

---

## 🚀 Quickstart & Interactive Walkthrough

### 1. Installation

```bash
# Clone repository
git clone https://github.com/your-username/arxiv-agent.git
cd arxiv-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration (Optional)

```bash
# Copy template
cp .env.example .env

# Set your Groq API key (optional — free tier at https://console.groq.com)
# If left blank, the agent runs in 100% offline extractive fallback mode!
nano .env
```

### 3. Usage Commands

<details open>
<summary><b>📖 Click to expand command examples</b></summary>

```bash
# 1. Digest by research topic (enters interactive QA loop automatically)
python main.py digest "KV-cache compression for LLMs"

# 2. Digest by explicit arXiv ID
python main.py digest 2401.12345

# 3. Digest by web URL
python main.py digest https://arxiv.org/abs/2401.12345

# 4. Generate briefing only and exit immediately (skip QA)
python main.py digest 2401.12345 --no-qa

# 5. Resume interactive QA session for an already processed paper
python main.py qa 2401.12345

# 6. View table of all saved local sessions
python main.py sessions

# 7. Run offline demo with bundled synthetic paper (no internet needed)
python main.py demo
```
</details>




## 📂 Project Directory Layout

```
arxiv-agent/
├── main.py                 # CLI entry point (argparse subcommands)
├── requirements.txt        # Pinned core dependencies
├── .env.example            # Environment template for GROQ_API_KEY
├── sample_data/
│   └── sample_paper.txt    # Synthetic paper for zero-network demo mode
├── sessions/               # Created at runtime: JSON session files per paper
├── chroma_store/           # Created at runtime: Persistent ChromaDB vector collections
└── src/
    ├── __init__.py         # Package initialization
    ├── config.py           # Centralized configuration & environment constants
    ├── models.py           # Core dataclasses: AgentState, PaperMetadata, Briefing, Chunk
    ├── arxiv_api.py        # Node 1 (Intent) & Node 2 (arXiv Search + Query Expansion)
    ├── pdf_parser.py       # Node 4 (Download + PyMuPDF/pypdf/OCR parsing chain)
    ├── chunking.py         # Node 5a (Reference section stripping & sliding window chunking)
    ├── embeddings.py       # Node 5b (Local BGE sentence-transformers with query prefix)
    ├── vectorstore.py      # ChromaDB client & cosine similarity retrieval
    ├── llm.py              # GroqProvider, MockProvider & AutoLLMProvider fallback
    ├── briefing.py         # Node 6 (Structured JSON briefing generation)
    ├── qa.py               # Node 7 (Grounded anti-hallucination RAG QA)
    ├── graph.py            # Orchestrator: 7-Node Agent State Graph runner
    ├── session.py          # Session persistence (save, load, list)
    └── cli.py              # Rich terminal interface (panels, spinners, tables)
```

---

<div align="center">
Thank you 
</div>
