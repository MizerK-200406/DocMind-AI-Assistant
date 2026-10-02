"""Định nghĩa lược đồ (Schemas) dữ liệu bằng Pydantic.

Module này chứa các lớp định nghĩa cấu trúc dữ liệu phản hồi, phục vụ tính năng
trích xuất thông tin có cấu trúc (Structured Outputs) từ văn bản tài liệu.
"""

from typing import Literal
from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    """Mô hình dữ liệu chứa thông tin được phân tích và trích xuất từ tài liệu.
    
    Attributes:
        title: Tiêu đề ngắn gọn của văn bản.
        doc_type: Phân loại tài liệu (báo cáo, hợp đồng, email, bài báo, khác).
        language: Ngôn ngữ chính (tiếng Việt, tiếng Anh, khác).
        summary: Tóm tắt nội dung văn bản từ 1-3 câu.
        key_points: Danh sách các ý chính quan trọng (tối thiểu 1, tối đa 5 ý).
        sentiment: Sắc thái cảm xúc chung của văn bản (tích cực, trung tính, tiêu cực).
        confidence: Điểm tin cậy của mô hình đánh giá (từ 0.0 đến 1.0).
    """
    title: str = Field(description="Tiêu đề ngắn gọn của văn bản")
    doc_type: Literal["bao_cao", "hop_dong", "email", "bai_bao", "khac"] = Field(description="Loại văn bản")
    language: Literal["vi", "en", "other"] = Field(description="Ngôn ngữ chính")
    summary: str = Field(description="Tóm tắt trong 1-3 câu")
    key_points: list[str] = Field(min_length=1, max_length=5, description="Danh sách các ý chính")
    sentiment: Literal["tich_cuc", "trung_tinh", "tieu_cuc"] = Field(description="Sắc thái chung của văn bản")
    confidence: float = Field(ge=0, le=1, description="Độ tự tin của mô hình, từ 0 đến 1")
