import sys
import time
from pathlib import Path

from app.config import settings
from app.ingest.chunker import chunk_document
from app.ingest.parser import parse_pdf
from app.rag import embedder, store

path = sys.argv[1]
source = Path(path).name

chunk_size = settings.CHUNK_SIZE
overlap = settings.CHUNK_OVERLAP

start = time.time()
chunks = chunk_document(parse_pdf(path), source, chunk_size, overlap)
print(f"Parse + chunk: {len(chunks)} chunk, {time.time() - start:.1f}s")

start = time.time()
vectors = embedder.embed_texts([c.text for c in chunks])
elapsed = time.time() - start
print(f"Embedding: {elapsed:.1f}s ({elapsed / len(chunks) * 1000:.1f} ms/chunk)")

store.delete_source(source)
store.add_chunks(chunks, vectors)
print(f"Đã lưu. Tổng số chunk trong DB: {store.get_collection().count()}")