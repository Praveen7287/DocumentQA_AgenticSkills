---
name: focused_summarizer
description: Generates concise, query-focused summaries from retrieved document chunks. Use when you need to answer specific questions based on document content rather than providing full document summaries.
metadata:
  category: summarization
  style: "abstractive"
  version: "1.0"
---
# Focused Summarization Instructions

## Prompt Engineering Template

CONTEXT: Document excerpts related to user query
TASK: Generate a concise answer focusing on the query
CONSTRAINTS:

Use only information from provided excerpts

Be specific and factual

Cite sources when possible

Acknowledge missing information

Keep under 3 paragraphs


## Techniques

1. **Extractive summarization**: Identify and combine key sentences
2. **Abstractive summarization**: Generate new sentences capturing essence
3. **Query-focused**: Filter information relevant to specific question
4. **Multi-document**: Synthesize information from multiple sources

## Quality Checks

- Faithfulness to source material
- Completeness for the query
- Conciseness and clarity
- Proper attribution