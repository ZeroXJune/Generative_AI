# Personal Assistant AI – A Generative RAG System

A Retrieval-Augmented Generation (RAG) application that allows users to upload documents, manage schedules, ask questions about their content, and receive intelligent notifications for upcoming deadlines.

## Features

- **Document Management**: Upload and index personal documents, notes, and schedules
- **Intelligent Q&A**: Ask questions about your documents and schedules using a fine-tuned LLM
- **Schedule Tracking**: Manage events and deadlines with automatic notifications
- **RAG Pipeline**: Text preprocessing → embeddings → vector search → contextual responses
- **Conversational Memory**: Multi-turn conversations maintaining context
- **Deployment-Ready**: Containerized with Docker for production deployment

## Architecture

```
User Input
    ↓
┌───────────────────────────────────────────────┐
│ Interface Layer (Streamlit/FastAPI)          │
└───────────────────┬───────────────────────────┘
                    ↓
┌───────────────────────────────────────────────┐
│ RAG Orchestration (LangChain/LlamaIndex)      │
│  - Prompt Engineering                         │
│  - Conversational Memory                      │
│  - Context Routing                            │
└───────────────────┬───────────────────────────┘
                    ↓
┌───────────────────────────────────────────────┐
│ Vector Database Layer (Chroma/FAISS)         │
│  - Document Retrieval                         │
│  - Similarity Search                          │
└───────────────────┬───────────────────────────┘
                    ↓
┌───────────────────────────────────────────────┐
│ Embedding Generation (HuggingFace Models)    │
│  - Text Encoding                              │
│  - Vector Representation                      │
└───────────────────┬───────────────────────────┘
                    ↓
┌───────────────────────────────────────────────┐
│ Data Pipeline (Preprocessing)                 │
│  - Text Cleaning                              │
│  - Tokenization                               │
│  - Chunking                                   │
└───────────────────────────────────────────────┘
```

## Project Structure

```
personal-assistant-ai/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── docker-compose.yml                 # Docker orchestration
├── Dockerfile                         # Container configuration
│
├── data/
│   ├── raw/                           # Original documents
│   │   ├── schedule_*.txt
│   │   ├── notes_*.txt
│   │   └── ...
│   └── processed/                     # Processed & embedded data
│       ├── chunks.jsonl
│       ├── embeddings.npy
│       └── index/
│
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── text_cleaner.py           # Text normalization
│   │   ├── tokenizer.py              # Tokenization logic
│   │   └── chunker.py                # Document chunking
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedding_generator.py    # Embedding creation
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py           # Vector DB management
│   │   └── retriever.py              # Context retrieval
│   │
│   ├── interface/
│   │   ├── __init__.py
│   │   ├── app.py                    # Streamlit/FastAPI app
│   │   └── utils.py                  # UI utilities
│   │
│   └── deployment/
│       ├── __init__.py
│       └── scheduler.py              # Notification scheduling
│
└── docs/
    ├── 01_Project_Proposal.md
    ├── 02_Prompt_Engineering.md
    ├── 03_Framework_Comparison.md
    ├── 04_Model_Customization.md
    └── 05_Final_Technical_Report.md
```

## Checkpoint Progress

- [x] **Checkpoint 1** (Weeks 1–4, Prelim, 20%): Foundation & Data Pipeline
  - Project proposal and problem framing
  - Raw dataset of 20–30 documents
  - Text preprocessing with before/after examples
  - Embedding generation and model justification
  - Repository setup and documentation

- [x] **Checkpoint 2** (Weeks 5–8, Midterm, 20%): Prompt Architecture & Vector Indexing
  - Four versioned system prompts with design rationale ([docs](docs/02_Prompt_Engineering.md))
  - Chat Completion API integration with an offline fallback
  - Chroma vector database, 26 documents → 150 indexed chunks
  - Distance metric comparison: Cosine vs Euclidean vs Dot product ([docs](docs/02_Vector_Indexing.md))

- [x] **Checkpoint 3** (Weeks 9–12, Semi-Final, 30%): RAG Orchestration & Application
  - Automated ingestion pipeline with content-hash change detection
  - LangChain integration via `Embeddings` and `BaseRetriever` adapters
  - Conversational memory with reference resolution before retrieval
  - Live demo, 7 queries ([docs](docs/04_RAG_Orchestration.md))

- [ ] **Checkpoint 4** (Weeks 13–16, Final, 30%): Deployment & Defense

## Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (for deployment)
- HuggingFace API key (optional, for hosted models)

### Installation

```bash
git clone <repo>
cd personal-assistant-ai
pip install -r requirements.txt
```

### Running the Data Pipeline

```bash
python src/preprocessing/text_cleaner.py --input data/raw/ --output data/processed/
python src/embeddings/embedding_generator.py --input data/processed/ --output data/processed/embeddings/
```

### Checkpoint 2: Vector Indexing & Retrieval

```bash
# Build the Chroma index from data/raw
python src/build_index.py

# Full demonstration: prompts, Chat API, retrieval, live queries
python src/checkpoint2_demo.py

# Reproduce the distance metric comparison
python src/experiments/metric_comparison.py
```

By default the Chat Completion client runs an offline extractive responder, so
the pipeline works with no credentials and no network. To answer with a real
model, point it at any OpenAI-compatible endpoint.

**Option A — Ollama (free, local, no API key).** Recommended for development
and for the live demo:

```bash
ollama serve                 # start the server
ollama pull llama3.2         # one-time model download

export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_MODEL="llama3.2"

python src/checkpoint2_demo.py
```

Ollama serves the same `/v1/chat/completions` contract as OpenAI, so this
needs no code change and no key. In Python, `ChatClient.for_ollama()` does the
same thing without environment variables.

**Option B — a hosted provider.** Requires a paid credential:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://..."    # optional: Azure, Groq, Together
export OPENAI_MODEL="gpt-3.5-turbo"
```

### Configuring credentials with a `.env` file

Rather than exporting variables in every shell, copy the template and fill it
in. `.env` is git-ignored, so a real key is never committed:

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY (or the Ollama settings)
```

The file is loaded automatically — no `export` required. Exported shell
variables take precedence over `.env`, so a one-off override still works.
Never put a real key in `.env.example`; that file **is** committed.

Verify what the client picked up before spending anything:

```bash
python -c "import sys; sys.path.insert(0,'src'); \
from llm.chat_client import ChatClient; print(ChatClient().get_info())"
```

`live: true` with `backend: openai` means a hosted call will be attempted;
`backend: offline` means the key was not found and the extractive responder
will answer instead.

`ChatClient.get_info()` reports which backend actually served a call
(`offline`, `local`, or `openai`), so a local run is never mistaken for a paid
API run.

Likewise, if `huggingface.co` is unreachable, `build_index.py` falls back to a
deterministic TF-IDF embedder and says so — no silent substitution.

### Checkpoint 3: Conversational RAG

```bash
python src/checkpoint3_demo.py     # ingestion + LangChain + memory + 7 queries
```

The ingestion pipeline is incremental — re-running it only processes documents
whose contents changed.

### Deadline Reminders

```bash
python src/reminders.py                      # what's due in the next 14 days
python src/reminders.py --days 30            # widen the horizon
python src/reminders.py --refresh            # re-scan data/raw first
python src/reminders.py --as-of 2026-09-15   # reproducible demo output
```

Implements the schedule-tracking objective from the proposal. Date arithmetic
is plain Python, not an LLM call, so alerts are exact and work offline — see
[docs/03_Schedule_Reminders.md](docs/03_Schedule_Reminders.md).

### Running the Application

```bash
streamlit run src/interface/app.py
```

### Running with Docker

```bash
docker-compose up --build
```

## Technologies Used

- **LLM Framework**: LangChain / LlamaIndex
- **Vector DB**: Chroma / FAISS
- **Embeddings**: Sentence-Transformers (HuggingFace)
- **Interface**: Streamlit / FastAPI
- **Deployment**: Docker & Docker Compose
- **Fine-tuning**: PEFT (LoRA/QLoRA)

## Author

Alber June Mumar - IS Professional Elective #4 Mini Capstone Project

## License

MIT
