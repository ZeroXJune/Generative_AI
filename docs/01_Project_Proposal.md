# Personal Assistant AI – Project Proposal

**Author**: Junie Mumar  
**Capstone Theme**: Personal/Domain Notes Assistant  
**Submission Date**: Checkpoint 1 (Prelim)  
**Course**: IS Professional Elective #4 – Generative AI Systems

---

## 1. Problem Statement

Modern knowledge workers manage scattered, unstructured information across multiple sources: notes, documents, schedules, research papers, and personal archives. Without a unified system, retrieving relevant information and maintaining awareness of upcoming deadlines becomes time-consuming and error-prone.

**Core Problem**: How can a generative AI system intelligently organize, retrieve, and reason over personal documents and schedules while proactively notifying users of time-sensitive items?

---

## 2. Target Users

- **Students and Academics**: Managing course materials, research notes, and assignment deadlines
- **Knowledge Workers**: Organizing project documentation, meeting notes, and schedules
- **Professionals**: Tracking domain-specific references and time-sensitive tasks
- **Anyone with Document Overload**: Needing intelligent search and schedule awareness

---

## 3. Chosen Theme & Scope

### Theme: Personal/Domain Notes Assistant with Schedule Integration

A RAG-powered chatbot that:
1. **Ingests** user-uploaded documents (notes, PDFs, schedules, forms)
2. **Indexes** content for intelligent retrieval via embeddings
3. **Answers** natural language questions grounded in the user's documents
4. **Tracks** schedules and deadlines with automatic notifications
5. **Maintains** conversation context across multi-turn interactions

### Why This Theme?

- **Real-world relevance**: Applies to students, professionals, and organizations
- **Clear RAG fit**: Documents provide grounded context, reducing hallucination
- **Measurable scope**: 20–30 documents sufficient for meaningful RAG performance
- **Feature richness**: Demonstrates full pipeline from ingestion to deployment
- **Notification component**: Adds practical value beyond typical Q&A bots

---

## 4. Dataset Description

### Dataset Source

A curated collection of **25–30 real and simulated personal documents**:

#### 4.1 Document Categories

| Category | Count | Content |
|----------|-------|---------|
| Academic Notes | 5 | Course notes, research summaries |
| Project Documentation | 4 | Project specs, design docs, meeting notes |
| Schedules & Calendars | 6 | Class schedules, event lists, deadlines |
| Personal References | 3 | Recipes, travel guides, how-to documents |
| Technical Guides | 4 | Setup instructions, API documentation |
| Meeting Minutes | 3 | Meeting summaries with action items |
| Task Lists | 2 | To-do lists with descriptions |
| Miscellaneous | 4 | Articles, reference materials |

**Total**: ~30 documents, ~50,000–100,000 tokens

#### 4.2 Data Format

- **Source**: .txt, .md, .pdf (sample), CSV exports
- **Size**: Mixed; some >2KB, some <500 bytes
- **Language**: English
- **Noise Level**: Moderate (realistic OCR artifacts, inconsistent formatting)

#### 4.3 Storage

- **Raw Data**: `/data/raw/` (version-controlled)
- **Access**: GitHub repository (all files <25MB)

---

## 5. Success Criteria

### Data Pipeline (Checkpoint 1)

✅ **Preprocessing Quality**
- Text cleaning removes 95%+ of artifacts (special chars, extra whitespace)
- Tokenization produces valid token sequences
- Before/after examples demonstrate effectiveness

✅ **Embedding Generation**
- All documents successfully embedded into vectors
- Embedding model justified (perplexity, coverage, alignment with task)
- Sample similarity search shows semantic relevance

✅ **Documentation**
- Clear README with setup instructions
- Organized repository with clean commit history
- Reflection on data challenges and mitigation

### Full Application (Final, Checkpoint 4)

✅ **Functional Requirements**
- Users can upload documents (multiple formats)
- Q&A system retrieves relevant context and answers accurately
- Schedule notifications trigger before deadlines
- Multi-turn conversations maintain history

✅ **Quality Metrics**
- RAG retrieval accuracy: ≥80% relevant context retrieved
- Hallucination rate: <10% of responses
- Response latency: <2 seconds per query
- Notification reliability: 100% on-time alerts

✅ **Non-Functional Requirements**
- Responsive UI (mobile + desktop)
- Containerized deployment (Docker)
- Supports 50+ documents without performance degradation
- Secure handling of user data (environment variables, no hardcoded secrets)

---

## 6. Technical Architecture Overview

### 6.1 Data Flow

```
User Upload (Documents/Schedules)
         ↓
    [Text Cleaner]
         ↓
    [Tokenizer]
         ↓
    [Chunker] → Chunks (512 tokens, 50% overlap)
         ↓
[Embedding Generator] → Dense Vectors (384–768 dims)
         ↓
   [Vector Store] → Indexed & Searchable
```

### 6.2 Query Flow

```
User Question
      ↓
[Embedding Generator] → Query Vector
      ↓
[Vector Store] → Top-K Retrieval (k=5)
      ↓
[Prompt + Context] → LLM Input
      ↓
[LLM] → Grounded Response
      ↓
[Notification Check] → Alert if deadline detected
      ↓
User Interface
```

### 6.3 Key Technologies

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Embeddings | Sentence-Transformers (MPNET) | Fast, high-quality, <1GB |
| Vector DB | Chroma (local) / Pinecone (cloud) | Simple, effective, scalable |
| LLM | OpenAI GPT-4 / Open-source Llama-2 | Production-grade, cost-effective |
| Orchestration | LangChain | Standard RAG framework, good docs |
| Interface | Streamlit | Rapid UI development, no frontend skills needed |
| Deployment | Docker | Reproducible, portable, cloud-ready |
| Fine-tuning | LoRA/QLoRA | Low VRAM, maintain base knowledge |

---

## 7. Checkpoint Roadmap

### Checkpoint 1: Foundation & Data Pipeline (Weeks 1–4)
- ✅ Finalize dataset (25–30 documents)
- ✅ Implement text preprocessing (cleaning, tokenization)
- ✅ Generate embeddings, justify model choice
- ✅ Document before/after examples
- ✅ Organized repository with README

### Checkpoint 2: Prompt Architecture & Vector Indexing (Weeks 5–8)
- Design 3+ structured system prompts
- Set up vector database with similarity search
- Compare distance metrics (Cosine, Euclidean, Dot Product)
- Integrate Chat Completion API

### Checkpoint 3: RAG Orchestration & Working Application (Weeks 9–12)
- Build automated ingestion pipeline
- Implement conversational memory
- Integrate LangChain/LlamaIndex orchestration
- Live demo with 5+ realistic queries

### Checkpoint 4: Deployment & Defense (Weeks 13–16)
- Fine-tuning analysis (LoRA/QLoRA)
- Web interface (Streamlit)
- Dockerization and CI/CD
- Final presentation and defense

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Documents too small for effective RAG | Use chunking with overlap; augment synthetic data |
| Hallucination in responses | Strict prompting; confidence thresholds; source attribution |
| Slow vector search at scale | Optimize chunk size; use approximate NN (FAISS) |
| API rate limits (OpenAI) | Cache common queries; use open-source LLM |
| Data privacy concerns | Local-first architecture; no cloud upload without consent |

---

## 9. Expected Outcomes

By project completion, this system will demonstrate:

1. **CILO 1**: Full data pipeline (cleaning → tokenization → embedding → indexing)
2. **CILO 2**: Scalable RAG with vector DB and orchestration framework
3. **CILO 3**: Open-source foundation model with PEFT fine-tuning understanding
4. **CILO 4**: Secure, containerized, deployed application

---

## 10. References & Resources

- **RAG Papers**: Lewis et al. (2020) – "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- **LangChain Docs**: https://python.langchain.com/
- **Sentence-Transformers**: https://www.sbert.net/
- **Chroma Vector DB**: https://www.trychroma.com/
- **Streamlit**: https://streamlit.io/

---

**Approved by**: Instructor (Jessie A. Melendres)  
**Date**: 2026-07-14
