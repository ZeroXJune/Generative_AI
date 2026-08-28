"""
RAG retriever: the glue between the vector index and the prompt layer.

Takes a natural-language question, embeds it with the same model used to
build the index, retrieves the nearest chunks from Chroma, assembles a
grounded prompt, and calls the Chat Completion API.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.chat_client import ChatClient, ChatResponse
from prompts.system_prompts import build_rag_messages, get_prompt
from retrieval.vector_store import VectorStore


class RAGRetriever:
    """Retrieve grounded context and answer questions over it."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder,
        chat_client: Optional[ChatClient] = None,
        top_k: int = 5,
        min_score: float = 0.0,
    ):
        """
        Wire the retrieval components together.

        Args:
            vector_store: An indexed VectorStore
            embedder: Any object exposing embed_texts(List[str]) -> np.ndarray;
                MUST be the same embedder used to build the index
            chat_client: Chat Completion client; created with defaults if omitted
            top_k: Number of chunks to retrieve per query
            min_score: Drop hits scoring below this threshold. Filtering weak
                context is what lets the grounded_qa prompt refuse instead of
                answering from irrelevant passages.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.chat_client = chat_client or ChatClient()
        self.top_k = top_k
        self.min_score = min_score

    def embed_query(self, question: str) -> np.ndarray:
        """
        Embed a single query string.

        Args:
            question: The query text

        Returns:
            A single embedding vector
        """
        return self.embedder.embed_texts([question])[0]

    def retrieve(
        self, question: str, top_k: Optional[int] = None, where: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve the chunks most relevant to a question.

        Args:
            question: The query text
            top_k: Override the default retrieval depth
            where: Optional Chroma metadata filter

        Returns:
            Ranked hits that clear `min_score`
        """
        hits = self.vector_store.query(
            self.embed_query(question), top_k=top_k or self.top_k, where=where
        )
        return [hit for hit in hits if hit["score"] >= self.min_score]

    def answer(
        self,
        question: str,
        prompt_name: str = "grounded_qa",
        history: Optional[List[Dict]] = None,
        **prompt_kwargs,
    ) -> Dict:
        """
        Run one full RAG turn: retrieve, assemble, generate.

        Args:
            question: The user's question
            prompt_name: Which system prompt to apply
            history: Prior conversation turns
            **prompt_kwargs: Placeholder values for the system prompt

        Returns:
            The answer, the supporting chunks, and call metadata
        """
        retrieved = self.retrieve(question)
        prompt = get_prompt(prompt_name)

        messages = build_rag_messages(
            question,
            retrieved,
            prompt_name=prompt_name,
            history=history,
            **prompt_kwargs,
        )
        response: ChatResponse = self.chat_client.complete(
            messages, temperature=prompt.temperature
        )

        return {
            "question": question,
            "answer": response.content,
            "sources": [
                {
                    "doc_id": hit["doc_id"],
                    "chunk_index": hit["chunk_index"],
                    "score": round(hit["score"], 4),
                }
                for hit in retrieved
            ],
            "num_retrieved": len(retrieved),
            "prompt": {"name": prompt.name, "version": prompt.version},
            "llm": response.summary(),
        }
