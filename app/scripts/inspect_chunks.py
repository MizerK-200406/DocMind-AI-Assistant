import sys
from pathlib import Path

from app.ingest.chunker import chunk_document
from app.ingest.parser import parse_pdf

path = sys.argv[1]
pages = parse_pdf(path)
print(f"Số trang có text: {len(pages)}")
if not pages:
    print("Không trích được text. Có thể đây là PDF scan, cần OCR.")
    sys.exit(1)

chunks = chunk_document(pages, source=Path(path).name)
lengths = [len(c.text) for c in chunks]
print(f"Số chunk: {len(chunks)}")
print(f"Độ dài: ngắn nhất {min(lengths)}, dài nhất {max(lengths)}, trung bình {sum(lengths) // len(lengths)}")

for c in chunks[:3]:
    print("-" * 60)
    print(f"[trang {c.page} | chunk {c.index} | {len(c.text)} ký tự]")
    print(c.text)