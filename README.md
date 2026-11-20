# CDC Health RAG Assistant

Retrieval-Augmented Generation (RAG) pipeline over CDC MMWR (Morbidity and Mortality Weekly Reports). Ask public health questions and get grounded, source-cited answers from real CDC documents.

## Architecture

```
CDC MMWR HTML corpus (scraped from cdc.gov/mmwr)
    └── Text chunking (800 tokens, 100 overlap)
        └── sentence-transformers (all-MiniLM-L6-v2) embeddings
            └── ChromaDB persistent vector store
                └── Semantic retrieval (top-k chunks)
                    └── Ollama (Llama 3 local LLM)
                        └── Grounded answer + source citations
                            └── FastAPI /ask endpoint
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull Llama 3 via Ollama (one-time)
ollama pull llama3

# 3. Scrape CDC MMWR reports
cd src && python scraper.py

# 4. Build vector store
python vectorstore.py

# 5. Run the API
cd ../api && uvicorn main:app --reload
```

## Example

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What does CDC report about COPD prevalence in rural communities?"}'
```

```json
{
  "answer": "According to MMWR Volume 73..., COPD prevalence is higher in rural counties...",
  "sources": ["mm7312", "mm7301"]
}
```

## Stack

Python · ChromaDB · sentence-transformers · Ollama · FastAPI · BeautifulSoup · Streamlit
