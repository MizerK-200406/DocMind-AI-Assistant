"""Module chia nhỏ văn bản (chunking) phục vụ RAG.

Sử dụng thuật toán phân tách đệ quy theo thứ tự ưu tiên ranh giới (đoạn -> dòng -> câu -> từ)
kết hợp kỹ thuật cửa sổ trượt (sliding window) bảo toàn độ gối (overlap) giữa các chunk liên tiếp.
"""

from dataclasses import dataclass

from app.ingest.parser import Page

# Thứ tự ưu tiên ranh giới cắt tự nhiên để tránh xé vụn câu/từ
SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    """Biểu diễn một đoạn văn bản sau khi cắt kèm thông tin nguồn."""
    text: str
    source: str
    page: int
    index: int


def _atomize(text: str, chunk_size: int, separators: list[str] = SEPARATORS) -> list[str]:
    """Phân rã văn bản đệ quy thành các mảnh nguyên tử (atoms) <= chunk_size.
    
    Ưu tiên tách ở ranh giới ngữ nghĩa lớn nhất trước (\n\n -> \n -> câu -> từ).
    """
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        # Khi đã duyệt hết các ký tự phân cách mà vẫn vượt kích thước, cắt cứng theo số ký tự
        return [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    sep, rest = separators[0], separators[1:]
    pieces = text.split(sep)
    if len(pieces) == 1:
        return _atomize(text, chunk_size, rest)
    
    atoms: list[str] = []
    for i, p in enumerate(pieces):
        piece = p + (sep if i < len(pieces) - 1 else "")
        if len(piece) > chunk_size:
            atoms.extend(_atomize(piece, chunk_size, rest))
        else:
            atoms.append(piece)   
    return atoms


def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Gộp các mảnh nguyên tử thành danh sách chunk hoàn chỉnh với độ gối (overlap).
    
    Khi tổng độ dài vượt chunk_size:
    1. Xuất chunk hiện tại.
    2. Thu nhỏ danh sách đệm bằng cách loại dần các atom đầu cho đến khi độ dài <= overlap
       nhằm làm phần gối đầu giữ ngữ cảnh cho chunk tiếp theo.
    """
    atoms = _atomize(text, chunk_size, SEPARATORS)
    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for atom in atoms:
        if length + len(atom) > chunk_size and current:
            chunks.append("".join(current).strip())
            # Giữ lại một phần cuối của chunk trước làm overlap cho chunk sau
            while current and (length > overlap or length + len(atom) > chunk_size):
                length -= len(current.pop(0))
        current.append(atom)
        length += len(atom)
    
    if current:
        chunks.append("".join(current).strip())
    return [c for c in chunks if c]


def chunk_document(pages: list[Page], source: str, chunk_size: int = 1000, overlap: int = 150) -> list[Chunk]:
    """Chia nhỏ toàn bộ trang tài liệu và đánh chỉ mục (index) tăng dần."""
    chunks: list[Chunk] = []
    for page in pages:
        for text in split_text(page.text, chunk_size, overlap):
            chunks.append(Chunk(text=text, source=source, page=page.page_number, index=len(chunks)))
    
    return chunks

    