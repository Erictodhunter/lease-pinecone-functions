# Lease Processing Pinecone Functions

Modal functions for storing and querying lease documents with Pinecone vector database.

## Setup

1. Create secrets in Modal:
```bash
modal secret create pinecone-secret PINECONE_API_KEY=your_key
modal secret create openai-secret OPENAI_API_KEY=your_key
