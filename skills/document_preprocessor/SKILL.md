---
name: document_preprocessor
description: Extracts text from documents (PDF, DOCX, TXT) and splits them into manageable chunks for processing. Use when you need to prepare documents for question answering, retrieval, or summarization tasks.
compatibility: Requires PyPDF2, python-docx libraries
metadata:
  category: document_processing
  max_chunk_size: "1000 tokens"
  version: "1.0"
---
# Document Preprocessing Instructions

## Step-by-Step Process

1. **Detect file format** from the file extension
2. **Extract text content** using appropriate library:
   - PDF: Use PyPDF2 with fallback for scanned documents
   - DOCX: Use python-docx preserving headings and structure
   - TXT/MD: Read with UTF-8 encoding
3. **Clean the text** by removing excessive whitespace and special characters
4. **Split into chunks** using paragraph boundaries first, then sentence boundaries
5. **Add metadata** to each chunk including source file and position

## Implementation Details

```python
def process_document(file_path):
    # Implementation here
    pass