# RAG Agent

A modular Retrieval-Augmented Generation (RAG) pipeline that combines semantic search, keyword retrieval, reranking, and LLM-based generation with guardrails and evaluation. Supports both cloud NVIDIA AI endpoints and local LLMs via Ollama.

## Architecture

```
PDF Document
    │
    ▼
┌─────────────────┐
│   Ingestion     │  PyPDFLoader → text cleaning
└────────┬────────┘
         ▼
┌─────────────────┐
│   Chunking      │  RecursiveCharacterTextSplitter (safety pre-split)
└────────┬────────┘          └→ SemanticChunker (NVIDIA embeddings, percentile)
         ▼
┌─────────────────┐
│   Indexing      │  Chroma (dense vector store) + BM25 (sparse keyword index)
└────────┬────────┘
         ▼
┌─────────────────┐
│   Retrieval     │  Multi-Query Retriever (query expansion) + BM25
└────────┬────────┘  Ensemble weighted 70/30
         ▼
┌──────────────────┐
│ Post-Processing  │  FlashRank reranker → context compression
└────────┬─────────┘
         ▼
┌─────────────────┐
│   Generation    │  LLM answer + source citations (traced via LangSmith)
└────────┬────────┘
         ▼
┌─────────────────┐
│   Guardrails    │  Banned phrase filter + citation validation
└────────┬────────┘
         ▼
    Final Answer
```

## Pipeline Components

| Component | File | Description |
|-----------|------|-------------|
| **Ingestion** | `components/ingestion.py` | Loads PDFs via `PyPDFLoader`, strips whitespace, filters empty pages |
| **Chunking** | `components/chunking.py` | Recursive pre-split (1500 chars, 150 overlap) → optional semantic split via NVIDIA embeddings |
| **Indexing** | `components/indexing.py` | Dual index: Chroma vector store (dense) + BM25 retriever (sparse) |
| **Retrieval** | `components/retrieval.py` | Hybrid ensemble: Multi-Query dense retriever (0.7 weight) + BM25 (0.3 weight) |
| **Post-Processing** | `components/post_processing.py` | FlashRank reranking via `ContextualCompressionRetriever` |
| **Generation** | `components/generation.py` | Prompt-based answer generation with `[Source, Page]` citations; traced via LangSmith `@traceable` |
| **Guardrails** | `components/guardrails.py` | Validates output: rejects AI identity phrases, ensures citation presence |
| **Evaluation** | `evals/ragas_eval.py` | RAGAS metrics: faithfulness, answer_relevancy, context_recall, context_precision |

## LLM Providers

| Provider | Config | Model |
|----------|--------|-------|
| **NVIDIA Cloud** | `config/llm_provider.py` | `meta/llama-3.3-70b-instruct` via NVIDIA AI Endpoints |
| **Local (Ollama/LM Studio)** | `config/llm_local.py` | Local OpenAI-compatible server (default: `localhost:8000/v1`) |

## Setup

### 1. Prerequisites

- Python ≥ 3.11
- [Ollama](https://ollama.com/download/) (for local mode)
- NVIDIA API key (for cloud mode)

### 2. Install

```bash
git clone https://github.com/kk-engineer/rag_agents.git
cd rag_agents

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

uv sync  # or: pip install -r requirements.txt
```

### 3. Environment Variables

```bash
export nvidia_api_key="nvapi-..."       # Required for NVIDIA AI endpoints
export langchain_api_key="lsv2_pt_..."  # Optional, for LangSmith tracing
```

### 4. Run

**Streamlit UI:**
```bash
streamlit run app.py
```

**Python pipeline (CLI):**
```bash
python app.py
```

## Usage Example

```python
from components.ingestion import DocumentIngestor
from components.chunking import SemanticProcessor
from components.indexing import IndexingEngine
from components.retrieval import HybridRetriever
from components.post_processing import ContextOptimizer
from components.generation import Generator
from components.guardrails import Guardrails

# 1. Ingest & chunk
raw_docs = DocumentIngestor("document.pdf").load_and_clean()
chunks = SemanticProcessor().split(raw_docs)

# 2. Index
indexer = IndexingEngine()
vector_store = indexer.build_vector_store(chunks)
bm25 = indexer.build_bm25_index(chunks)

# 3. Retrieve & optimize
retriever = HybridRetriever(vector_store, bm25, llm=llm)
optimizer = ContextOptimizer(retriever.ensemble)
docs = optimizer.get_optimized_docs("Your question here?")

# 4. Generate with guardrails
answer = Generator(llm).generate("Your question?", docs)
is_valid, result = Guardrails.validate_output(answer)
```

## Evaluation

RAGAS metrics are computed automatically when running `python app.py`:

- **Faithfulness** — Is the answer factually grounded in the context?
- **Answer Relevancy** — How relevant is the answer to the question?
- **Context Recall** — Does the retrieved context contain all needed information?
- **Context Precision** — How relevant are the retrieved documents?

Define test cases in `app.py` and run:
```bash
python app.py
```

## Project Structure

```
rag_agent/
├── app.py                    # Streamlit app + RAG pipeline orchestrator
├── pyproject.toml            # Project config & dependencies
├── components/
│   ├── ingestion.py          # PDF loading & cleaning
│   ├── chunking.py           # Semantic text splitting
│   ├── indexing.py           # Chroma + BM25 index building
│   ├── retrieval.py          # Hybrid ensemble retriever
│   ├── post_processing.py    # FlashRank reranking
│   ├── generation.py         # LLM answer generation
│   └── guardrails.py         # Output validation
├── config/
│   ├── llm_provider.py       # NVIDIA AI endpoint wrapper
│   └── llm_local.py          # Local LLM server wrapper
├── evals/
│   ├── ragas_eval.py         # RAGAS evaluation suite
│   └── evaluator.py          # Simple keyword-based evaluator
└── db/                       # Chroma persisted vector store
```
