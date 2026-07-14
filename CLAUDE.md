# Claude Code Project Configuration

## Project Overview

**Personal Assistant AI – Generative RAG Capstone Project**

A Retrieval-Augmented Generation system that allows users to upload documents, ask intelligent questions, and receive notifications about schedules and deadlines.

**Course**: IS Professional Elective #4 – Generative AI Systems  
**Instructor**: Jessie A. Melendres  
**Capstone Theme**: Personal/Domain Notes Assistant  
**Branch**: `claude/personal-assistant-ai-yxy5sf`

## Quick Start

### Prerequisites
- Python 3.10+
- pip (Python package manager)

### Installation

```bash
# Clone repo (already done)
cd /home/user/TrikRide_App

# Install dependencies
pip install -r requirements.txt

# Run text preprocessing demo
python src/preprocessing/text_cleaner.py

# Run chunking demo
python src/preprocessing/chunker.py

# Run embedding demo (requires sentence-transformers download ~50MB)
python src/embeddings/embedding_generator.py
```

### Full Pipeline Execution

```bash
# Run complete data pipeline
python src/pipeline.py

# This will:
# 1. Load 25 documents from data/raw/
# 2. Clean and preprocess text
# 3. Chunk documents (512 tokens, 50% overlap)
# 4. Generate embeddings (384-dim vectors)
# 5. Save processed data to data/processed/
```

## Project Structure

```
personal-assistant-ai/
├── README.md                          # Main documentation
├── CLAUDE.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git configuration
│
├── data/
│   ├── raw/                           # 25 input documents (text files)
│   └── processed/                     # Output after pipeline (generated)
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py                    # Main orchestration script
│   │
│   ├── preprocessing/
│   │   ├── text_cleaner.py            # Text cleaning & normalization
│   │   └── chunker.py                 # Document chunking
│   │
│   ├── embeddings/
│   │   └── embedding_generator.py     # Embedding generation
│   │
│   ├── retrieval/                     # Checkpoint 2 (vector DB)
│   ├── interface/                     # Checkpoint 4 (web UI)
│   └── deployment/                    # Checkpoint 4 (Docker)
│
└── docs/
    ├── 01_Project_Proposal.md         # Problem statement, theme, dataset
    └── CHECKPOINT1_REFLECTION.md      # Data challenges & solutions
```

## Checkpoint Progress

### Checkpoint 1: Foundation & Data Pipeline ✅
**Status**: Complete and committed  
**Due**: August 15, 2026  
**Components**:
- [x] Project proposal (1-2 pages)
- [x] Dataset (25 documents)
- [x] Text preprocessing with before/after examples
- [x] Embedding generation with model justification
- [x] Repository documentation and setup

**Key Files**:
- `docs/01_Project_Proposal.md` – Complete problem statement
- `docs/CHECKPOINT1_REFLECTION.md` – Reflection on data challenges
- `src/preprocessing/` – Text cleaning and chunking
- `src/embeddings/embedding_generator.py` – Embedding creation

### Checkpoint 2: Prompt Architecture & Vector Indexing ⏳
**Due**: September 19, 2026  
**Components**:
- Prompt engineering document (3+ system prompts)
- API integration demo (Chat Completion API)
- Vector database setup (Chroma/FAISS)
- Distance metric comparison (Cosine, Euclidean, Dot Product)

**Next Steps**:
1. Create `docs/02_Prompt_Engineering.md`
2. Implement `src/retrieval/vector_store.py` (Chroma integration)
3. Create `src/retrieval/retriever.py` (similarity search)
4. Set up LangChain/LlamaIndex orchestration

### Checkpoint 3: RAG Orchestration ⏳
**Due**: October 17, 2026  
**Components**:
- Automated ingestion pipeline
- RAG application (LangChain/LlamaIndex)
- Conversational memory
- Live demo with 5+ queries

### Checkpoint 4: Deployment & Defense ⏳
**Due**: November 14, 2026  
**Components**:
- Fine-tuning analysis (LoRA/QLoRA)
- Web interface (Streamlit)
- Containerization (Docker)
- Deployment evidence
- Final presentation & defense

## Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Embeddings** | Sentence-Transformers | Fast, <1GB, high quality (384-dim) |
| **Vector DB** | Chroma (local) / FAISS (scale) | Simple for learning, scalable |
| **LLM Framework** | LangChain | Standard, good docs, RAG-focused |
| **LLM** | GPT-3.5-Turbo / Llama-2 | Production-grade, cost-effective |
| **Interface** | Streamlit | Rapid development, no frontend skills |
| **Deployment** | Docker | Reproducible, portable, cloud-ready |
| **Fine-tuning** | PEFT (LoRA/QLoRA) | Low VRAM, maintains base knowledge |

## Data Pipeline Overview

```
Raw Documents (25 files, .txt)
           ↓
   [TextCleaner]
   - Remove HTML/URLs
   - Normalize Unicode
   - Remove special chars
   - Remove extra whitespace
           ↓
    [Tokenizer]
    - Split into tokens
    - Count tokens per doc
           ↓
    [Chunker]
    - 512 token chunks
    - 50% overlap
    - Preserve metadata
           ↓
[EmbeddingGenerator]
    - all-MiniLM-L6-v2
    - 384-dimensional vectors
    - Batch processing
           ↓
Processed Data
    - chunks.jsonl (with embeddings)
    - cleaning_examples.json
    - Pipeline statistics
```

## Code Quality Standards

- ✅ Type hints on all functions
- ✅ Docstrings for classes/methods
- ✅ Example scripts demonstrating functionality
- ✅ No hardcoded paths or secrets
- ✅ Error handling for edge cases
- ✅ Modular, testable components

## Running Examples

### Text Cleaning Example
```bash
python src/preprocessing/text_cleaner.py
# Output: Before/after examples with reduction metrics
```

### Chunking Example
```bash
python src/preprocessing/chunker.py
# Output: Sample chunks with token counts and boundaries
```

### Embedding Example
```bash
python src/embeddings/embedding_generator.py
# Output: Embedding statistics, similarity examples
# Downloads model (~50MB) on first run
```

### Full Pipeline
```bash
python src/pipeline.py
# Output: 
# - Loads 25 documents
# - Generates embeddings for all chunks
# - Saves to data/processed/chunks.jsonl
# - Prints statistics and timing
```

## Environment Variables

Optional configuration:
```bash
export EMBEDDING_MODEL="all-MiniLM-L6-v2"  # Default embedding model
export CHUNK_SIZE="512"                     # Tokens per chunk
export CHUNK_OVERLAP="100"                  # Token overlap
```

For production (Checkpoint 4+):
```bash
export OPENAI_API_KEY="sk-..."              # ChatGPT API key
export VECTOR_DB_PATH="/data/vector_store"  # Vector DB location
export LOG_LEVEL="INFO"                     # Logging verbosity
```

## Common Tasks

### Add a new document
```bash
# Save as .txt file in data/raw/
cp my_document.txt data/raw/

# Rerun pipeline
python src/pipeline.py
```

### Test a preprocessing function
```python
from src.preprocessing.text_cleaner import TextCleaner

cleaner = TextCleaner()
cleaned = cleaner.clean("Your text here!!!")
print(cleaned)
```

### Batch embed documents
```python
from src.embeddings.embedding_generator import EmbeddingGenerator

gen = EmbeddingGenerator()
embeddings = gen.embed_texts(["doc1", "doc2", "doc3"])
print(embeddings.shape)  # (3, 384)
```

## Debugging Tips

### Pipeline fails on import
```bash
pip install -r requirements.txt --force-reinstall
```

### Unicode decode errors
```bash
# Files must be UTF-8 encoded
file -i data/raw/*.txt  # Check encoding
iconv -f ISO-8859-1 -t UTF-8 input.txt > output.txt  # Convert if needed
```

### Embedding model download fails
```bash
# Manually download to cache
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
# Or use offline mode
export HF_DATASETS_OFFLINE=1
```

### Out of memory
```bash
# Reduce batch size
python src/pipeline.py --batch-size 8
```

## Useful Resources

- **LangChain Docs**: https://python.langchain.com/
- **Sentence-Transformers**: https://www.sbert.net/
- **Chroma Vector DB**: https://www.trychroma.com/
- **HuggingFace Models**: https://huggingface.co/models
- **Capstone Guide**: docs/01_Project_Proposal.md

## Contact & Support

**Course**: IS Professional Elective #4 – Generative AI Systems  
**Instructor**: Jessie A. Melendres  
**Office Hours**: Thursday 4:00-5:30 PM (Room 201)

For technical issues or clarifications on project requirements, please reach out during office hours or via email.

---

**Last Updated**: July 14, 2026  
**Version**: 0.1.0 (Checkpoint 1)
