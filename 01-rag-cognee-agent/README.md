# RAG Cognee Agent

A Retrieval-Augmented Generation (RAG) project that combines vector search (retrieval) with an LLM to generate context-aware answers.

## What it does
- Ingests documents and creates embeddings
- Retrieves the most relevant chunks for a user query
- Uses the retrieved context to generate a grounded response

## Tech stack
- Python
- Cognee (RAG / retrieval pipeline)
- Embeddings (local/fast embedding setup)
- LLM via API (API keys are not included in this repo)

## How to run
1. Create a virtual environment
2. Install dependencies (see project files / requirements if available)
3. Create `.env` from `.env.example` and set your API key
4. Run the main script / notebook in this folder

## Notes
- Large files and secrets are excluded from the repository.
