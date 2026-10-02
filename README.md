# 🧠 DocMind - Trợ Lý AI & Trích Xuất Tài Liệu Thông Minh

DocMind là một dịch vụ backend FastAPI hiệu năng cao, tích hợp mô hình ngôn ngữ lớn (DeepSeek thông qua NVIDIA NIM API) để cung cấp khả năng trò chuyện theo ngữ cảnh lịch sử, truyền phát thời gian thực (Streaming SSE) và trích xuất thông tin văn bản có cấu trúc chuẩn xác.

---

## 🚀 Tính Năng Nổi Bật

- **Hội thoại thông minh (Contextual Chat):** Tự động ghi nhớ và chắt lọc lịch sử tin nhắn gần nhất đưa vào ngữ cảnh của mô hình.
- **Truyền phát thời gian thực (Server-Sent Events - SSE):** Trả về từng token văn bản ngay khi sinh ra, mang lại trải nghiệm mượt mà cho người dùng.
- **Lưu trữ an toàn (Atomic Transaction SQLite):** Ghi nhận đồng thời cặp câu hỏi - câu trả lời trong một giao dịch nguyên tử, chống phân mảnh hay ô nhiễm dữ liệu khi xảy ra lỗi.
- **Tự động thử lại thông minh (Resilience & Retry):** Sử dụng `tenacity` với chiến lược Exponential Backoff để vượt qua các lỗi mạng tạm thời hoặc chạm ngưỡng Rate-Limit của API.
- **Trích xuất thông tin có cấu trúc (Structured Extraction):** Tích hợp Pydantic Schema và vòng lặp tự sửa lỗi (Self-Correction Loop) khi LLM phản hồi sai cấu trúc JSON.
- **Đóng gói Docker hoàn chỉnh:** Sẵn sàng triển khai với Dockerfile tối ưu và Docker Compose có cấu hình Persistent Volume cho database.
- **Kiểm thử tự động (Test Suite):** Bộ unit test và integration test toàn diện bằng `pytest` và `FastAPI TestClient`.

---

## 📂 Cấu Trúc Dự Án

```text
DocMind/
├── app/
│   ├── __init__.py
│   ├── main.py          # Ứng dụng FastAPI, định tuyến và xử lý ngoại lệ
│   ├── db.py            # Quản lý kết nối SQLite, schema và giao dịch
│   ├── llm.py           # Kết nối NVIDIA NIM API, cơ chế retry tự động
│   ├── schemas.py       # Pydantic schemas cho dữ liệu trích xuất
│   └── structured.py    # Logic trích xuất JSON và vòng lặp tự sửa lỗi
├── tests/
│   ├── conftest.py      # Fixtures phục vụ kiểm thử (fake LLM, SQLite in-memory/tmp)
│   └── test_api.py      # Bộ kiểm thử tự động các endpoints
├── Dockerfile           # Đóng gói image container Python 3.12-slim
├── docker-compose.yml   # Điều phối dịch vụ và gắn kết volume lưu trữ
├── requirements.txt     # Danh sách các thư viện phụ thuộc
├── .dockerignore        # Loại bỏ các file rác khi build image
└── .gitignore           # Bảo vệ các file bí mật (.env, .venv, *.db)
```

---

## 🛠️ Yêu Cầu Hệ Thống

- **Python:** 3.11 hoặc 3.12+
- **Docker & Docker Compose** (Tùy chọn nếu muốn chạy qua container)
- **NVIDIA API Key** (Đăng ký miễn phí tại [NVIDIA NIM](https://build.nvidia.com/))

---

## ⚙️ Hướng Dẫn Cài Đặt & Chạy Cục Bộ

### 1. Khởi tạo môi trường ảo và cài đặt thư viện
```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo
# Trên Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Trên Linux/macOS:
source .venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Cấu hình biến môi trường
Tạo file `.env` tại thư mục gốc với nội dung:
```ini
NVIDIA_API_KEY=nvapi-your-key-here
```

### 3. Khởi động ứng dụng
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Truy cập tài liệu API trực quan tại: **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 🐳 Hướng Dẫn Chạy Bằng Docker

Chỉ cần một lệnh duy nhất với Docker Compose:

```bash
# Khởi chạy container
docker compose up --build -d

# Xem logs hoạt động
docker compose logs -f

# Dừng dịch vụ
docker compose down
```

---

## 🧪 Chạy Kiểm Thử (Testing)

Dự án đi kèm bộ kiểm thử tự động không tốn quota API nhờ hệ thống giả lập LLM:

```bash
python -m pytest -v
```

---

## 📡 Danh Sách API Endpoints Chính

| Phương thức | Đường dẫn | Mô tả |
|---|---|---|
| `GET` | `/health` | Kiểm tra trạng thái hoạt động của server |
| `POST` | `/chat` | Trò chuyện với trợ lý AI (đồng bộ) |
| `POST` | `/chat/stream` | Trò chuyện thời gian thực qua Server-Sent Events |
| `GET` | `/conversations/{conv_id}` | Lấy toàn bộ lịch sử tin nhắn của một cuộc hội thoại |
| `POST` | `/extract` | Trích xuất thông tin cấu trúc từ văn bản tự do |
