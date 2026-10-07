"""Script dòng lệnh nạp và nhúng tài liệu PDF vào cơ sở dữ liệu vector ChromaDB.

Cách dùng:
    python -m app.scripts.ingest "duong/dan/toi/file.pdf"

Quy trình thực thi:
1. Parse PDF & làm sạch text (bỏ mục lục TOC, bỏ header/footer lặp mép trang).
2. Phân rã thành các chunk ngữ nghĩa có gối đầu (sliding window overlap).
3. Tạo vector embeddings thông qua mô hình BGE-M3.
4. Xóa dữ liệu cũ của file (nếu có) và lưu các chunk mới vào ChromaDB.
"""

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

# Giai đoạn 1: Parse PDF và cắt nhỏ thành các chunk ngữ nghĩa
start = time.time()
chunks = chunk_document(parse_pdf(path), source, chunk_size, overlap)
print(f"Parse + chunk: {len(chunks)} chunk, {time.time() - start:.1f}s")

# Giai đoạn 2: Tạo vector embeddings cho toàn bộ các chunk
start = time.time()
vectors = embedder.embed_texts([c.text for c in chunks])
elapsed = time.time() - start
print(f"Embedding: {elapsed:.1f}s ({elapsed / len(chunks) * 1000:.1f} ms/chunk)")

# Giai đoạn 3: Lưu trữ bền vững vào ChromaDB
store.delete_source(source)
store.add_chunks(chunks, vectors)
print(f"Đã lưu. Tổng số chunk trong DB: {store.get_collection().count()}")