"""Build and query a ChromaDB vector store from scraped MMWR text."""
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import hashlib

DATA_DIR = Path(__file__).parent.parent / "data" / "mmwr"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma"

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION  = "cdc_mmwr"
CHUNK_SIZE  = 800
CHUNK_OVER  = 100


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVER) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
        i += size - overlap
    return chunks


def build_vectorstore(force_rebuild: bool = False):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL)

    if force_rebuild:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass

    col = client.get_or_create_collection(COLLECTION, embedding_function=ef)
    existing = set(col.get()["ids"]) if col.count() > 0 else set()

    docs, ids, metas = [], [], []
    for txt_file in sorted(DATA_DIR.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        for j, chunk in enumerate(_chunk_text(text)):
            doc_id = hashlib.md5(f"{txt_file.stem}_{j}".encode()).hexdigest()
            if doc_id in existing:
                continue
            docs.append(chunk)
            ids.append(doc_id)
            metas.append({"source": txt_file.stem, "chunk": j})

    if docs:
        col.add(documents=docs, ids=ids, metadatas=metas)
        print(f"Added {len(docs)} chunks. Total in store: {col.count()}")
    else:
        print(f"Vector store up to date. Total chunks: {col.count()}")
    return col


def query(question: str, top_k: int = 5) -> list[dict]:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL)
    col = client.get_collection(COLLECTION, embedding_function=ef)
    results = col.query(query_texts=[question], n_results=top_k)
    out = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        out.append({"text": doc, "source": meta["source"],
                    "chunk": meta["chunk"], "distance": dist})
    return out


if __name__ == "__main__":
    build_vectorstore()
    hits = query("What are the COPD prevalence trends in the United States?")
    for h in hits:
        print(f"\n[{h['source']}]  dist={h['distance']:.3f}\n{h['text'][:300]}...")
