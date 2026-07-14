from typing import List, Dict, Tuple


class DocumentChunker:
    """Split documents into overlapping chunks for embedding."""

    def __init__(self, chunk_size: int = 512, overlap: int = 100):
        """
        Initialize chunker.

        Args:
            chunk_size: Target size of each chunk in tokens (approximate)
            overlap: Number of overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, doc_id: str = "unknown") -> List[Dict]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            doc_id: Document identifier for metadata

        Returns:
            List of chunk dictionaries with metadata
        """
        tokens = text.split()
        chunks = []

        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_tokens = tokens[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_tokens)

            if len(chunk_text.strip()) > 0:
                chunks.append(
                    {
                        "text": chunk_text,
                        "doc_id": doc_id,
                        "chunk_index": len(chunks),
                        "start_token": i,
                        "end_token": i + len(chunk_tokens),
                        "token_count": len(chunk_tokens),
                    }
                )

        return chunks

    def chunk_documents(
        self, documents: List[Tuple[str, str]]
    ) -> List[Dict]:
        """
        Chunk multiple documents.

        Args:
            documents: List of (text, doc_id) tuples

        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        for text, doc_id in documents:
            chunks = self.chunk_text(text, doc_id)
            all_chunks.extend(chunks)
        return all_chunks

    def get_chunking_example(self, text: str, doc_id: str = "example") -> dict:
        """
        Get example of how text is chunked.

        Args:
            text: Text to chunk
            doc_id: Document ID

        Returns:
            Dictionary with chunking statistics and samples
        """
        chunks = self.chunk_text(text, doc_id)
        tokens = text.split()

        return {
            "total_tokens": len(tokens),
            "total_chunks": len(chunks),
            "avg_chunk_size": round(len(tokens) / len(chunks), 1) if chunks else 0,
            "chunks": [
                {
                    "index": i,
                    "text_preview": c["text"][:100] + "...",
                    "token_count": c["token_count"],
                }
                for i, c in enumerate(chunks[:3])
            ],
        }


if __name__ == "__main__":
    chunker = DocumentChunker(chunk_size=100, overlap=20)

    # Example text
    sample_text = """
    RETRIEVAL-AUGMENTED GENERATION (RAG) - DEEP DIVE

    What is RAG?
    RAG (Retrieval-Augmented Generation) combines two components:
    1. Retriever: Fetches relevant documents/passages from a knowledge base
    2. Generator: Uses retrieved context + user query to generate responses

    Benefits over vanilla LLM:
    - Grounded responses (cites sources)
    - Reduced hallucination
    - Updates without retraining (just add new documents)
    - Transparent reasoning (show which documents were used)

    RAG Pipeline
    1. USER QUERY - Input: "What is the capstone deadline?"
    2. QUERY ENCODING - Convert query to embedding using same model as documents
    3. RETRIEVAL - Search vector database for similar document chunks
    4. CONTEXT ASSEMBLY - Combine retrieved chunks into context string
    5. PROMPT ENGINEERING - Create prompt with context and query
    6. GENERATION - Pass to LLM and get response
    """

    example = chunker.get_chunking_example(sample_text, "rag_overview")

    print("CHUNKING EXAMPLE:")
    print(f"Total tokens: {example['total_tokens']}")
    print(f"Total chunks: {example['total_chunks']}")
    print(f"Average chunk size: {example['avg_chunk_size']} tokens")
    print("\nFirst 3 chunks:")
    for chunk in example["chunks"]:
        print(f"\n  Chunk {chunk['index']} ({chunk['token_count']} tokens):")
        print(f"  {chunk['text_preview']}")
