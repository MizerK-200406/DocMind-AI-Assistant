"""Module trích xuất và tiền xử lý nội dung văn bản từ tệp PDF.

Sử dụng PyMuPDF (fitz) để đọc text và chuẩn hóa bảng mã Unicode tiếng Việt dựng sẵn (NFC),
loại bỏ các ký tự ẩn và khoảng trắng thừa trước khi chuyển sang bước chunking.
"""

import re
import unicodedata
from dataclasses import dataclass

import pymupdf


@dataclass
class Page:
    """Biểu diễn dữ liệu văn bản của một trang PDF."""
    page_number: int
    text: str


def clean_text(text: str) -> str:
    """Tiền xử lý và làm sạch văn bản thô trích xuất từ PDF.
    
    - Chuẩn hóa NFC: Chuyển Unicode tổ hợp (NFD - thường gặp trong PDF) về dạng dựng sẵn (NFC)
      để đảm bảo tính đồng nhất khi tìm kiếm từ khóa và embedding tiếng Việt.
    - Xóa soft-hyphen (\\u00ad): Ký tự gạch nối mềm ẩn dùng để ngắt dòng trong PDF.
    - Thu gọn khoảng trắng và giới hạn dòng trống liên tiếp tối đa là 2.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(path: str) -> list[Page]:
    """Đọc file PDF từ đường dẫn và trả về danh sách các trang chứa văn bản hợp lệ."""
    pages: list[Page] = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = clean_text(page.get_text())
            if text:
                pages.append(Page(page_number=i, text=text))
    return pages

