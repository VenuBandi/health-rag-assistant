"""FastAPI endpoint for CDC MMWR RAG assistant."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import ask

app = FastAPI(title="CDC MMWR Health RAG Assistant", version="1.0.0")


class Question(BaseModel):
    question: str
    top_k: int = 5
    model: str = "llama3"

class Answer(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask_endpoint(q: Question):
    if not q.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty")
    result = ask(q.question, top_k=q.top_k, model=q.model)
    return Answer(answer=result["answer"], sources=result["sources"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
