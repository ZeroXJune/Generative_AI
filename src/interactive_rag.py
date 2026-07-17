"""
Interactive RAG Demo - Test semantic search from command line
"""

import json
import numpy as np
from pathlib import Path

def cosine_similarity(a, B):
    """Compute cosine similarity between vector a and vectors in B."""
    a = np.array(a)
    B = np.array(B)
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return np.dot(B_norm, a_norm)

def load_chunks(chunks_file="data/processed/chunks.jsonl"):
    """Load chunks with embeddings."""
    chunks = []
    with open(chunks_file, 'r') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def search(query, chunks, top_k=3, min_similarity=0.3):
    """
    Search for similar chunks to query.
    For demo purposes, use embedding of closest matching chunk as query embedding.
    """
    # Find chunk most similar to query text (simple heuristic)
    query_lower = query.lower()
    best_match_idx = 0
    best_match_score = 0

    for idx, chunk in enumerate(chunks):
        # Count keyword matches in text
        text_lower = chunk['text'].lower()
        query_words = query_lower.split()
        matches = sum(1 for word in query_words if word in text_lower)
        if matches > best_match_score:
            best_match_score = matches
            best_match_idx = idx

    # Use best matching chunk's embedding as query embedding
    query_embedding = chunks[best_match_idx]['embedding']

    # Get embeddings for all chunks
    embeddings = np.array([chunk["embedding"] for chunk in chunks])

    # Compute similarities
    similarities = cosine_similarity(query_embedding, embeddings)

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Filter by threshold
    results = []
    for idx in top_indices:
        if similarities[idx] >= min_similarity:
            results.append({
                "chunk": chunks[idx],
                "similarity": float(similarities[idx])
            })

    return results

def main():
    """Interactive RAG demo."""
    print("\n" + "="*60)
    print("🤖 PERSONAL ASSISTANT AI - Interactive RAG Demo")
    print("="*60)

    # Load chunks
    print("\n📚 Loading SQuAD dataset...")
    chunks = load_chunks()
    print(f"   ✓ Loaded {len(chunks)} chunks")
    print(f"   ✓ Embedding dimension: {len(chunks[0]['embedding'])}")

    print("\n" + "-"*60)
    print("📝 Enter your questions (type 'exit' to quit)")
    print("   Example: 'What is artificial intelligence?'")
    print("-"*60)

    while True:
        print()
        query = input("🔍 Question: ").strip()

        if query.lower() == 'exit':
            print("\n✅ Goodbye!")
            break

        if not query:
            print("   Please enter a question.")
            continue

        # Search
        results = search(query, chunks, top_k=3, min_similarity=0.0)

        if not results:
            print("   ❌ No relevant passages found.")
            continue

        # Display results
        print(f"\n   Found {len(results)} relevant passage(s):\n")

        for i, result in enumerate(results, 1):
            chunk = result['chunk']
            similarity = result['similarity']

            print(f"   📄 Result {i} (Similarity: {similarity:.1%})")
            print(f"      Source: {chunk['doc_id']}")
            print(f"      Chunk: {chunk['chunk_index']}")
            print(f"      Tokens: {chunk['token_count']}")
            print(f"\n      Text: {chunk['text'][:300]}...")
            print()

if __name__ == "__main__":
    main()
