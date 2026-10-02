"""Module trích xuất dữ liệu có cấu trúc từ văn bản tự do (Structured Information Extraction).

Sử dụng LLM kết hợp với Pydantic JSON Schema và cơ chế Vòng lặp Tự sửa lỗi (Self-Correction Loop):
Nếu LLM trả về chuỗi JSON không khớp với Schema hoặc sai cú pháp, hệ thống sẽ tự động gửi lại thông báo
lỗi chi tiết của Pydantic cho LLM để mô hình tự sửa và thử lại.
"""

import json
import logging
import re

from pydantic import BaseModel, ValidationError

from app import llm

logger = logging.getLogger("docmind.structured")


def _clean_json(raw: str) -> str:
    """Làm sạch chuỗi JSON trả về từ LLM.
    
    Loại bỏ các ký tự bọc markdown như ```json ... ``` và bất kỳ văn bản thừa nào
    bên ngoài cặp ngoặc nhọn { ... } đầu tiên và cuối cùng.
    
    Args:
        raw: Chuỗi văn bản thô trả về từ mô hình LLM.
        
    Returns:
        str: Chuỗi JSON tinh khiết đã được cắt tỉa.
    """
    raw = raw.strip()
    # Xóa markdown code block ở đầu và cuối chuỗi nếu có
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    # Tìm vị trí bắt đầu và kết thúc của khối JSON { ... }
    start, end = raw.find("{"), raw.rfind("}")
    return raw[start:end + 1] if start != -1 and end != -1 else raw


def extract(
    text: str,
    schema_cls: type[BaseModel],
    instruction: str,
    max_retries: int = 2,
) -> BaseModel:
    """Trích xuất thông tin từ văn bản tự do theo định dạng Pydantic Schema định sẵn.
    
    Cơ chế hoạt động:
    1. Chuyển đổi Pydantic model thành định dạng chuẩn JSON Schema.
    2. Gửi lời nhắc (Prompt) yêu cầu LLM phản hồi theo đúng JSON Schema đã cung cấp.
    3. Kiểm tra tính hợp lệ bằng schema_cls.model_validate_json.
    4. Nếu vi phạm (ValidationError), gửi lại đoạn JSON lỗi cùng thông báo chi tiết
       của Pydantic cho LLM để sửa lỗi (Self-Correction Loop).
    
    Args:
        text: Nội dung văn bản cần phân tích / trích xuất.
        schema_cls: Lớp Pydantic Model dùng làm khuôn mẫu dữ liệu (vd: DocumentInfo).
        instruction: Chỉ thị hệ thống định hướng ngữ cảnh cho LLM.
        max_retries: Số lần thử lại tối đa khi LLM trả sai cấu trúc JSON (mặc định 2 lần).
        
    Returns:
        BaseModel: Đối tượng Pydantic đã được điền dữ liệu hợp lệ.
        
    Raises:
        ValueError: Ném ra khi vượt quá số lần thử lại mà LLM vẫn không tạo được JSON hợp lệ.
    """
    schema = json.dumps(schema_cls.model_json_schema(), ensure_ascii=False)
    system = (
        f"{instruction}\n\n"
        "Chỉ trả về MỘT đối tượng JSON hợp lệ theo JSON Schema sau, "
        f"Không kèm theo bất kỳ lời giải thích hay cú pháp markdown nào:\n{schema}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]

    last_error = None
    for attempt in range(max_retries + 1):
        # Đặt temperature=0 để kết quả trích xuất mang tính xác định cao nhất
        resp = llm.chat_completion(
            messages=messages,
            temperature=0,
        )
        raw = resp.choices[0].message.content or ""
        try:
            return schema_cls.model_validate_json(_clean_json(raw))
        except ValidationError as e:
            last_error = e
            logger.warning("[extract] Lần thử %d thất bại: %d lỗi xác thực schema", attempt + 1, e.error_count())
            # Lưu lại câu trả lời lỗi của assistant và thêm lời nhắc sửa lỗi từ user
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"JSON chưa hợp lệ:\n{e}\nHãy sửa lại và chỉ trả về JSON hợp lệ duy nhất.",
            })

    raise ValueError(f"LLM không trả về JSON hợp lệ sau {max_retries + 1} lần thử: {last_error}")
