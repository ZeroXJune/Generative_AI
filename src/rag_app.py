"""
Conversational RAG application.

Checkpoint 3 deliverable: the orchestration layer that ties every previous
component into one multi-turn application.

    question
       │
       ▼
  ConversationMemory.resolve()      "when is it due?" -> "when is it due? Checkpoint 3"
       │
       ▼
  embed + Chroma search             the Checkpoint 2 index, unchanged
       │
       ▼
  build_rag_messages(history=...)   grounded prompt + prior turns
       │
       ▼
  ChatClient                        hosted API, local Ollama, or offline
       │
       ▼
  answer + citations  ->  recorded back into memory
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from build_index import load_embedder
from ingestion.pipeline import IngestionPipeline, IngestionReport
from llm.chat_client import ChatClient
from memory.conversation import ConversationMemory
from prompts.system_prompts import build_rag_messages, get_prompt
from retrieval.vector_store import VectorStore

DEFAULT_COLLECTION = "personal_assistant_cp3"
DEFAULT_PERSIST_DIR = "data/vector_store_cp3"


class RAGApplication:
    """A conversational RAG assistant over the user's own documents."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        raw_dir: str = "data/raw",
        top_k: int = 4,
        min_score: float = 0.0,
        max_turns: int = 6,
        chat_client: Optional[ChatClient] = None,
        reset: bool = False,
    ):
        """
        Wire the application together.

        Args:
            collection_name: Chroma collection to read and write
            persist_directory: On-disk index location
            raw_dir: Source document directory
            top_k: Chunks retrieved per question
            min_score: Drop weaker hits; what makes refusal reachable
            max_turns: Conversation turns retained
            chat_client: Chat Completion client; built with defaults if omitted
            reset: Rebuild the collection from scratch
        """
        self.embedder, self.using_real_model = load_embedder()
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            space="cosine",
            reset=reset,
        )
        self.ingestion = IngestionPipeline(
            vector_store=self.vector_store, embedder=self.embedder, raw_dir=raw_dir
        )
        self.memory = ConversationMemory(max_turns=max_turns)
        self.chat_client = chat_client or ChatClient()
        self.top_k = top_k
        self.min_score = min_score

    def ingest(self, force: bool = False) -> IngestionReport:
        """
        Bring the index up to date with the corpus.

        Args:
            force: Re-ingest every document regardless of the manifest

        Returns:
            A report describing what changed
        """
        return self.ingestion.ingest(force=force)

    def as_langchain_retriever(self, top_k: Optional[int] = None):
        """
        Expose this application's index as a LangChain retriever.

        Args:
            top_k: Override retrieval depth

        Returns:
            A PersonalAssistantRetriever usable in any LangChain chain
        """
        from integrations.langchain_adapters import PersonalAssistantRetriever

        return PersonalAssistantRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            top_k=top_k or self.top_k,
            min_score=self.min_score,
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve chunks for an already-resolved query.

        Args:
            query: A standalone search query
            top_k: Override retrieval depth

        Returns:
            Ranked hits clearing `min_score`
        """
        vector = self.embedder.embed_texts([query])[0]
        hits = self.vector_store.query(vector, top_k=top_k or self.top_k)
        return [hit for hit in hits if hit["score"] >= self.min_score]

    def ask(self, question: str, prompt_name: str = "grounded_qa") -> Dict:
        """
        Answer one conversational turn.

        Resolves references against the conversation, retrieves, generates, and
        records the exchange so the next follow-up can be resolved in turn.

        Args:
            question: The user's question, as asked
            prompt_name: Which system prompt to apply

        Returns:
            The answer plus retrieval and generation metadata
        """
        resolved = self.memory.resolve(question, self.chat_client)
        retrieved = self.retrieve(resolved)
        prompt = get_prompt(prompt_name)

        messages = build_rag_messages(
            question,
            retrieved,
            prompt_name=prompt_name,
            history=self.memory.as_messages(),
        )
        response = self.chat_client.complete(messages, temperature=prompt.temperature)

        sources = [hit["doc_id"] for hit in retrieved]
        self.memory.add_turn(
            question=question,
            answer=response.content,
            resolved_query=resolved,
            sources=sources,
        )

        return {
            "question": question,
            "resolved_query": resolved,
            "was_rewritten": resolved != question,
            "answer": response.content,
            "sources": [
                {"doc_id": hit["doc_id"], "score": round(hit["score"], 4)}
                for hit in retrieved
            ],
            "num_retrieved": len(retrieved),
            "turn": self.memory.total_turns,
            "llm": response.summary(),
        }

    def reset_conversation(self) -> None:
        """Clear conversation history without touching the index."""
        self.memory.clear()

    def get_info(self) -> dict:
        """Return a snapshot of the application's configuration and state."""
        return {
            "indexed_chunks": self.vector_store.count(),
            "indexed_documents": len(self.vector_store.list_documents()),
            "embedder": self.embedder.model_name,
            "using_real_model": self.using_real_model,
            "distance_space": self.vector_store.space,
            "llm_backend": self.chat_client.get_info()["backend"],
            "memory": self.memory.summary(),
        }
