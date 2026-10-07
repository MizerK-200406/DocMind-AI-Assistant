import chromadb

from app.config import settings
from app.ingest.chunker import Chunk

CHROMA_PATH = settings.CHROMA_PATH
COLLECTION = settings.CHROMA_COLLECTION

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

def list_sources() -> list[str]:
    data = get_collection().get(include=["metadatas"])
    return sorted({m["source"] for m in data["metadatas"]})

def delete_source(source: str):
    get_collection().delete(where={"source": source})

def add_chunks(chunks: list[Chunk], embeddings: list[list[float]], batch: int = 500):
    col = get_collection()
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        col.upsert(
            ids=[f"{c.source}:{c.index}" for c in part],
            embeddings=embeddings[i : i + batch],
            documents=[c.text for c in part],
            metadatas=[{"source": c.source, "page": c.page, "page_label": c.label, "index": c.index} for c in part],
        )


def search(query_embedding: list[float], k: int = 5, source: str | None = None) -> list[dict]:
    res = get_collection().query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"source": source} if source else None,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "page_label": meta.get("page_label", str(meta["page"])),
            "score": 1 - dist,
        }
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]