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

- [ ] **Checkpoint 2** (Weeks 5–8, Midterm, 20%): Prompt Architecture & Vector Indexing
- [ ] **Checkpoint 3** (Weeks 9–12, Semi-Final, 30%): RAG Orchestration & Application
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
