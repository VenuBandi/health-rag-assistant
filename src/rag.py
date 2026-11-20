"""RAG pipeline: ChromaDB retrieval + Ollama LLM generation."""
import ollama
from vectorstore import query

DEFAULT_MODEL = "llama3"
SYSTEM_PROMPT = (
    "You are a public health research assistant. Answer the user's question "
    "using ONLY the provided CDC MMWR context. "
    "Cite the source document for each claim. "
    "If the context does not contain enough information, say so clearly."
)


def build_prompt(question: str, context_chunks: list[dict]) -> str:
    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )
    return f"""Context from CDC MMWR reports:
{context_text}

Question: {question}

Answer based on the context above:"""


def ask(question: str, top_k: int = 5, model: str = DEFAULT_MODEL) -> dict:
    chunks = query(question, top_k=top_k)
    if not chunks:
        return {"answer": "No relevant CDC documents found.", "sources": []}

    prompt = build_prompt(question, chunks)

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    answer = response["message"]["content"]
    sources = list({c["source"] for c in chunks})
    return {"answer": answer, "sources": sources, "context_chunks": chunks}


if __name__ == "__main__":
    result = ask("What does CDC report about chronic disease trends in rural communities?")
    print(result["answer"])
    print("\nSources:", result["sources"])
