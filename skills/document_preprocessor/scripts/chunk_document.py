#!/usr/bin/env python3
"""
Document Chunker - Split documents into manageable chunks
"""

import argparse
import re
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size  # in tokens
        self.overlap = overlap
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def chunk_by_tokens(self, text: str) -> List[Dict]:
        """Split text into chunks by token count"""
        words = word_tokenize(text)
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[max(0, start - self.overlap):end]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'start_word': max(0, start - self.overlap),
                'end_word': end,
                'chunk_id': len(chunks)
            })
            
            start = end - self.overlap
        
        return chunks
    
    def chunk_by_sentences(self, text: str) -> List[Dict]:
        """Split text into chunks by sentences"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = len(word_tokenize(sentence))
            
            if current_length + sentence_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'start_sentence': i - len(current_chunk),
                    'end_sentence': i - 1,
                    'chunk_id': len(chunks)
                })
                # Keep overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    s_tokens = len(word_tokenize(s))
                    if overlap_length + s_tokens <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += s_tokens
                    else:
                        break
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_tokens
        
        # Add last chunk
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'start_sentence': len(sentences) - len(current_chunk),
                'end_sentence': len(sentences) - 1,
                'chunk_id': len(chunks)
            })
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> List[Dict]:
        """Split text into chunks by paragraphs"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, para in enumerate(paragraphs):
            para_tokens = len(word_tokenize(para))
            
            if current_length + para_tokens > self.chunk_size and current_chunk:
                chunks.append({
                    'text': '\n\n'.join(current_chunk),
                    'start_para': i - len(current_chunk),
                    'end_para': i - 1,
                    'chunk_id': len(chunks)
                })
                
                # Keep last paragraph for overlap
                current_chunk = [current_chunk[-1]] if current_chunk else []
                current_length = len(word_tokenize(current_chunk[0])) if current_chunk else 0
            
            current_chunk.append(para)
            current_length += para_tokens
        
        if current_chunk:
            chunks.append({
                'text': '\n\n'.join(current_chunk),
                'start_para': len(paragraphs) - len(current_chunk),
                'end_para': len(paragraphs) - 1,
                'chunk_id': len(chunks)
            })
        
        return chunks

def main():
    parser = argparse.ArgumentParser(description="Chunk documents for processing")
    parser.add_argument("--input", required=True, help="Input text file")
    parser.add_argument("--method", choices=['tokens', 'sentences', 'paragraphs'], 
                       default='paragraphs', help="Chunking method")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in tokens")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap between chunks")
    parser.add_argument("--output", help="Output JSON file path")
    
    args = parser.parse_args()
    
    # Read input file
    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Initialize chunker
    chunker = DocumentChunker(chunk_size=args.chunk_size, overlap=args.overlap)
    
    # Chunk based on method
    if args.method == 'tokens':
        chunks = chunker.chunk_by_tokens(text)
    elif args.method == 'sentences':
        chunks = chunker.chunk_by_sentences(text)
    else:  # paragraphs
        chunks = chunker.chunk_by_paragraphs(text)
    
    # Output results
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2)
        print(f"Created {len(chunks)} chunks, saved to {args.output}")
    else:
        for chunk in chunks:
            print(f"\n=== Chunk {chunk['chunk_id']} ===")
            print(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])
            print()

if __name__ == "__main__":
    main()