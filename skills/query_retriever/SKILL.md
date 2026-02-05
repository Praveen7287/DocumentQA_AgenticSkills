---
name: query_retriever
description: Finds document sections most relevant to a user query using semantic similarity search. Use when you need to locate specific information within processed documents for answering questions.
compatibility: Requires sentence-transformers, numpy
metadata:
  category: information_retrieval
  embedding_model: "all-MiniLM-L6-v2"
  version: "1.0"
---
# Query Retrieval Instructions

## Retrieval Methodology

1. **Embed document chunks** using sentence-transformers model
2. **Embed user query** using same model
3. **Calculate cosine similarity** between query and chunk embeddings
4. **Return top-k results** with similarity scores above threshold

## Advanced Features

- **Hybrid search**: Combine semantic and keyword matching
- **Query expansion**: Add synonyms and related terms
- **Reranking**: Use cross-encoder for better precision
- **Caching**: Store embeddings to avoid recomputation

## Implementation

```python
def retrieve_relevant_chunks(query, chunks, top_k=5):
    # Implementation here
    pass