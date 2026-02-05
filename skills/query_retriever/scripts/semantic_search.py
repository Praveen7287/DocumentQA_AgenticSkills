#!/usr/bin/env python3
"""
Semantic Search for Document Retrieval
Uses sentence transformers for embedding-based search
"""

import numpy as np
from typing import List, Dict, Tuple
import argparse
import json
from pathlib import Path

class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic search with sentence transformer model
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.model = None
        self.documents = []
        self.embeddings = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            raise ImportError(
                "Please install sentence-transformers: pip install sentence-transformers"
            )
    
    def index_documents(self, documents: List[str]):
        """
        Index documents by creating embeddings
        
        Args:
            documents: List of document texts to index
        """
        self.documents = documents
        print(f"Creating embeddings for {len(documents)} documents...")
        self.embeddings = self.model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        print("Embeddings created successfully.")
    
    def search(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity score (0-1)
        
        Returns:
            List of dictionaries with document index, text, and similarity score
        """
        if self.embeddings is None:
            raise ValueError("No documents indexed. Call index_documents first.")
        
        # Encode query
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Calculate similarity scores (cosine similarity)
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] >= threshold:
                results.append({
                    'index': int(idx),
                    'text': self.documents[idx],
                    'score': float(similarities[idx]),
                    'metadata': {
                        'length_chars': len(self.documents[idx]),
                        'length_words': len(self.documents[idx].split())
                    }
                })
        
        return results
    
    def hybrid_search(self, query: str, documents: List[str], 
                     top_k: int = 5, alpha: float = 0.7) -> List[Dict]:
        """
        Hybrid search combining semantic and keyword matching
        
        Args:
            query: Search query
            documents: List of documents to search
            top_k: Number of results to return
            alpha: Weight for semantic vs keyword (0-1, where 1 is semantic only)
        """
        # Semantic scores
        semantic_results = self.search(query, top_k=len(documents))
        semantic_scores = {r['index']: r['score'] for r in semantic_results}
        
        # Keyword scores (simple term frequency)
        query_terms = set(query.lower().split())
        keyword_scores = {}
        
        for i, doc in enumerate(documents):
            doc_terms = set(doc.lower().split())
            common_terms = query_terms.intersection(doc_terms)
            if query_terms:
                keyword_scores[i] = len(common_terms) / len(query_terms)
            else:
                keyword_scores[i] = 0
        
        # Normalize scores
        max_semantic = max(semantic_scores.values()) if semantic_scores else 1
        max_keyword = max(keyword_scores.values()) if keyword_scores else 1
        
        # Combine scores
        combined_scores = {}
        for i in range(len(documents)):
            semantic = semantic_scores.get(i, 0) / max_semantic if max_semantic > 0 else 0
            keyword = keyword_scores.get(i, 0) / max_keyword if max_keyword > 0 else 0
            combined_scores[i] = alpha * semantic + (1 - alpha) * keyword
        
        # Get top-k combined results
        top_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for idx, score in top_indices:
            if score > 0:
                results.append({
                    'index': int(idx),
                    'text': documents[idx],
                    'score': float(score),
                    'semantic_score': float(semantic_scores.get(idx, 0)),
                    'keyword_score': float(keyword_scores.get(idx, 0))
                })
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Semantic search for documents")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--documents", help="JSON file with documents array")
    parser.add_argument("--text-file", help="Text file with one document per line")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results")
    parser.add_argument("--method", choices=['semantic', 'hybrid'], default='semantic')
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Model name")
    
    args = parser.parse_args()
    
    # Load documents
    documents = []
    if args.documents:
        with open(args.documents, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                documents = data
            else:
                documents = [data]
    elif args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            documents = [line.strip() for line in f if line.strip()]
    
    if not documents:
        print("No documents provided. Use --documents or --text-file")
        return
    
    # Initialize search
    search = SemanticSearch(model_name=args.model)
    search.index_documents(documents)
    
    # Perform search
    if args.method == 'semantic':
        results = search.search(args.query, top_k=args.top_k)
    else:
        results = search.hybrid_search(args.query, documents, top_k=args.top_k)
    
    # Display results
    print(f"\nQuery: {args.query}")
    print(f"Found {len(results)} relevant documents:\n")
    
    for i, result in enumerate(results, 1):
        print(f"Result {i} (Score: {result['score']:.3f}):")
        print(f"Text: {result['text'][:200]}..." if len(result['text']) > 200 else result['text'])
        if 'semantic_score' in result:
            print(f"  Semantic: {result['semantic_score']:.3f}, Keyword: {result['keyword_score']:.3f}")
        print()

if __name__ == "__main__":
    main()