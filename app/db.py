"""Module quản lý cơ sở dữ liệu SQLite cho DocMind.

Cung cấp các hàm khởi tạo schema, quản lý phiên hội thoại (conversations)
và lưu trữ lịch sử tin nhắn (messages) với cơ chế giao dịch an toàn (atomic transactions).
"""

import os
import sqlite3
import uuid
from contextlib import contextmanager

# Đường dẫn file SQLite: ưu tiên đọc từ biến môi trường DB_PATH (dùng cho Docker volume /data)
DB_PATH = os.getenv("DB_PATH", "docmind.db")


@contextmanager
def get_conn():
    """Context manager quản lý kết nối và giao dịch (transaction) với SQLite.
    
    - Tự động gán row_factory = sqlite3.Row để truy cập cột theo tên (dict-like).
    - Tự động gọi conn.commit() khi hoàn thành khối lệnh an toàn.
    - Luôn đảm bảo đóng kết nối conn.close() trong khối finally.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Khởi tạo cấu trúc các bảng dữ liệu nếu chưa tồn tại.
    
    Tạo 2 bảng chính:
    1. conversations: Lưu danh danh sách ID các cuộc hội thoại.
    2. messages: Lưu từng tin nhắn (user/assistant) kèm khóa ngoại liên kết tới conversation.
    3. Index idx_messages_conv: Tối ưu tốc độ truy vấn lịch sử tin nhắn theo conversation_id.
    """
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conv
                ON messages(conversation_id, id);
            """
        )


def create_conversation() -> str:
    """Tạo một phiên hội thoại mới với UUID duy nhất.
    
    Returns:
        str: Mã định danh hội thoại (conv_id) vừa tạo.
    """
    conv_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO conversations(id) VALUES(?)",
            (conv_id,)
        )
        return conv_id


def conversation_exists(conv_id: str) -> bool:
    """Kiểm tra xem một hội thoại có tồn tại trong hệ thống hay không.
    
    Args:
        conv_id: ID cuộc hội thoại cần kiểm tra.
        
    Returns:
        bool: True nếu tồn tại, False nếu không tìm thấy.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM conversations WHERE id = ?",
            (conv_id,)
        ).fetchone()
        return row is not None


def add_message(conv_id: str, role: str, content: str):
    """Ghi một tin nhắn đơn lẻ vào cơ sở dữ liệu.
    
    Args:
        conv_id: ID cuộc hội thoại.
        role: Vai trò người gửi ('user' hoặc 'assistant').
        content: Nội dung tin nhắn dạng chuỗi.
    """
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conv_id, role, content),
        )


def get_messages(conv_id: str, limit: int | None = None) -> list[dict]:
    """Lấy danh sách tin nhắn của một cuộc hội thoại.
    
    Args:
        conv_id: ID cuộc hội thoại.
        limit: Số lượng tin nhắn gần nhất cần lấy (nếu None thì lấy toàn bộ theo thứ tự thời gian tăng dần).
        
    Returns:
        list[dict]: Danh sách các tin nhắn dưới dạng [{'role': '...', 'content': '...'}].
    """
    with get_conn() as conn:
        if limit is not None:
            # Lấy N tin nhắn mới nhất (ORDER BY id DESC), sau đó đảo ngược lại để giữ đúng trình tự thời gian
            rows = conn.execute(
                """
                SELECT role, content FROM messages
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conv_id, limit),
            ).fetchall()
            rows = list(reversed(rows))
        else:
            rows = conn.execute(
                """
                SELECT role, content FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (conv_id,),
            ).fetchall()
    return [dict(row) for row in rows]


def add_exchange(conv_id: str, user_content: str, assistant_content: str):
    """Ghi cặp câu hỏi (user) và câu trả lời (assistant) trong cùng một transaction nguyên tử (atomic).
    
    Cơ chế này đảm bảo tính nhất quán: hoặc cả hai tin nhắn đều được lưu thành công,
    hoặc không có gì được lưu (ví dụ nếu có lỗi xảy ra giữa chừng). Tránh hiện tượng
    lưu câu hỏi mồ côi khi mô hình AI gặp lỗi phản hồi.
    
    Args:
        conv_id: ID cuộc hội thoại.
        user_content: Nội dung tin nhắn của người dùng.
        assistant_content: Nội dung phản hồi của trợ lý AI.
    """
    with get_conn() as conn:
        conn.executemany(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            [(conv_id, "user", user_content), (conv_id, "assistant", assistant_content)],
        )
