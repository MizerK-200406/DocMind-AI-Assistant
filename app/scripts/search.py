"""Công cụ dòng lệnh tìm kiếm ngữ nghĩa nhanh trong ChromaDB.

Cách dùng:
    python -m app.scripts.search "nội dung cần tìm"
"""

import sys

from app.config import settings
from app.rag import embedder, store

query = " ".join(sys.argv[1:])
vector = embedder.embed_texts([query])[0]

for rank, hit in enumerate(store.search(vector, k=settings.RAG_TOP_K), start=1):
    print(f"#{rank}  score={hit['score']:.3f}  trang {hit['page']}")
    print(hit["text"][:300].replace("\n", " "))
    print()