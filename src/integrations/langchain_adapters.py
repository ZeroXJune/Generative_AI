"""
LangChain interop.

Checkpoint 3 asks for a RAG application built with LangChain or LlamaIndex.
Rather than rebuild the retrieval stack a second time inside LangChain -- and
discard the measured Checkpoint 2 work in the process -- this module exposes
the existing components *through* LangChain's interfaces.

Two adapters do the whole job:

  * `PersonalAssistantEmbeddings` implements `langchain_core.embeddings.Embeddings`,
    so our embedder can drive any LangChain vector store.
  * `PersonalAssistantRetriever` implements `langchain_core.retrievers.BaseRetriever`,
    so our Chroma index can be dropped into any LangChain chain.

The benefit is that the cosine-vs-Euclidean analysis, the tuned chunking, and
the offline fallbacks all still apply -- LangChain composes them rather than
replacing them.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever


def to_documents(hits: List[Dict]) -> List[Document]:
    """
    Convert VectorStore results into LangChain Documents.

    Args:
        hits: Result dictionaries from VectorStore.query

    Returns:
        LangChain Documents carrying the retrieval metadata
    """
    return [
        Document(
            page_content=hit["text"],
            metadata={
                "doc_id": hit.get("doc_id", "unknown"),
                "chunk_index": hit.get("chunk_index", -1),
                "score": hit.get("score", 0.0),
                "distance": hit.get("distance", 0.0),
                "rank": hit.get("rank", 0),
            },
        )
        for hit in hits
    ]


class PersonalAssistantEmbeddings(Embeddings):
    """Adapts this project's embedders to the LangChain Embeddings interface."""

    def __init__(self, embedder):
        """
        Args:
            embedder: Any object exposing embed_texts(List[str]) -> ndarray.
                Works with MiniLM, the lexical fallback, or the mock.
        """
        self.embedder = embedder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of documents.

        Args:
            texts: Document texts

        Returns:
            One embedding per input, as plain lists
        """
        return [vector.tolist() for vector in self.embedder.embed_texts(texts)]

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Args:
            text: The query string

        Returns:
            The query embedding as a plain list
        """
        return self.embedder.embed_texts([text])[0].tolist()


class PersonalAssistantRetriever(BaseRetriever):
    """
    Adapts this project's Chroma store to the LangChain retriever interface.

    Any LangChain component that accepts a retriever accepts this, which means
    the Checkpoint 2 index -- cosine space, 120/30 chunking, metadata for
    citations -- is reused rather than rebuilt.
    """

    vector_store: Any
    embedder: Any
    top_k: int = 5
    min_score: float = 0.0

    # BaseRetriever is a Pydantic model; the project's store and embedder are
    # plain classes, so arbitrary attribute types must be permitted.
    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Retrieve documents for a query.

        Called by LangChain; use `.invoke(query)` rather than calling directly.

        Args:
            query: The search query
            run_manager: LangChain callback manager, supplied by the framework

        Returns:
            Matching Documents that clear `min_score`
        """
        query_vector = self.embedder.embed_texts([query])[0]
        hits = self.vector_store.query(query_vector, top_k=self.top_k)
        return to_documents([h for h in hits if h["score"] >= self.min_score])
