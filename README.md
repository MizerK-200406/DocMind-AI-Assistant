# 🧠 DocMind - Trợ Lý AI, Trích Xuất Dữ Liệu & RAG Tài Liệu Thông Minh

DocMind là dịch vụ backend hiệu năng cao xây dựng trên nền tảng **FastAPI**, tích hợp mô hình ngôn ngữ lớn (thông qua NVIDIA NIM API), cơ sở dữ liệu vector **ChromaDB**, và mô hình nhúng đa ngôn ngữ **BAAI/bge-M3**. Hệ thống cung cấp khả năng trò chuyện ngữ cảnh, truyền phát thời gian thực (Streaming SSE), trích xuất dữ liệu có cấu trúc chuẩn xác và pipeline **Hỏi đáp trên tài liệu (RAG)** chuyên sâu kèm cơ chế chống ảo giác.

---

## 🚀 Tính Năng Nổi Bật

- **Hỏi đáp tài liệu thông minh (RAG Pipeline):** Tìm kiếm ngữ nghĩa (Semantic Search) trên các tài liệu PDF đã nạp, tổng hợp câu trả lời ngắn gọn và chính xác.
- **Trích dẫn nguồn & Số trang (Citations & Attribution):** Tự động đính kèm số thứ tự nguồn `[1]`, `[2]`, số trang gốc trong PDF, đoạn trích và điểm tương đồng (similarity score).
- **Chống ảo giác (Rejection Guardrail):** Tự động kiểm tra ngưỡng tin cậy `MIN_SCORE` — nếu tài liệu không chứa đủ thông tin liên quan, hệ thống từ chối trả lời mà không tốn quota gọi LLM.
- **Tiền xử lý tài liệu thông minh (Smart PDF Ingestion):** Tự động phát hiện và loại bỏ trang Mục Lục (TOC), làm sạch header/footer lặp mép trang, chuẩn hóa unicode NFC.
- **Phân rã văn bản đệ quy (Recursive Text Splitting):** Chia nhỏ văn bản theo thứ tự ưu tiên ngữ nghĩa (đoạn ➔ dòng ➔ câu ➔ từ) kết hợp cửa sổ trượt gối đầu (sliding window overlap) bảo toàn ngữ cảnh.
- **Mô hình nhúng hiện đại (Dense Embeddings):** Sử dụng `BAAI/bge-M3`, tự động nhận diện GPU/CUDA và kích hoạt FP16 nửa độ chính xác để tăng tốc xử lý.
- **Hội thoại theo ngữ cảnh (Contextual Chat):** Tự động chọn lọc và đưa các tin nhắn gần nhất vào context window của LLM.
- **Truyền phát thời gian thực (Server-Sent Events - SSE):** Trả về từng token văn bản ngay khi sinh ra cho trải nghiệm phản hồi mượt mà.
- **Lưu trữ an toàn (Atomic Transaction SQLite):** Ghi nhận tin nhắn và phiên chat trong một giao dịch nguyên tử, ngăn ngừa ô nhiễm dữ liệu khi xảy ra sự cố.
- **Khả năng phục hồi cao (Resilience & Retry):** Sử dụng `tenacity` với Exponential Backoff để vượt qua các lỗi mạng tạm thời hoặc chạm ngưỡng Rate-Limit.
- **Trích xuất dữ liệu có cấu trúc (Structured Extraction):** Tích hợp Pydantic Schema và vòng lặp tự sửa lỗi (Self-Correction Loop) khi LLM phản hồi sai cấu trúc JSON.
- **Quản lý cấu hình tập trung (Centralized Settings):** Gom toàn bộ tham số hệ thống vào một nơi duy nhất với giá trị mặc định rõ ràng.
- **Kiểm thử tự động toàn diện (Test Suite):** Bộ test bao phủ toàn bộ API, parser, chunker, config và pipeline RAG với 100% test pass.

---

## 📂 Cấu Trúc Dự Án

```text
DocMind/
├── app/
│   ├── __init__.py
│   ├── config.py            # Quản lý cấu hình tập trung toàn bộ biến môi trường
│   ├── db.py                # Quản lý kết nối SQLite, schema và giao dịch
│   ├── llm.py               # Kết nối NVIDIA NIM API, cơ chế retry tự động
│   ├── main.py              # Ứng dụng FastAPI, định tuyến và xử lý ngoại lệ
│   ├── schemas.py           # Pydantic schemas cho dữ liệu trích xuất
│   ├── structured.py        # Logic trích xuất JSON và vòng lặp tự sửa lỗi
│   ├── eval/                # Bộ câu hỏi đánh giá benchmark chất lượng truy xuất
│   │   └── questions.jsonl  # Tập dữ liệu câu hỏi và trang chuẩn
│   ├── ingest/              # Tiền xử lý PDF và chia nhỏ văn bản
│   │   ├── __init__.py
│   │   ├── chunker.py       # Phân rã văn bản đệ quy có độ gối (overlap)
│   │   └── parser.py        # Trích xuất PDF, lọc TOC và header/footer lặp
│   ├── rag/                 # Module hỏi đáp trên tài liệu
│   │   ├── __init__.py
│   │   ├── embedder.py      # Nhúng vector với mô hình BAAI/bge-M3 (hỗ trợ GPU/FP16)
│   │   ├── pipeline.py      # Điều phối luồng RAG 5 bước và guardrail lọc ngưỡng
│   │   └── store.py         # Quản lý vector store ChromaDB (cosine space)
│   └── scripts/             # Các công cụ dòng lệnh (CLI Tools)
│       ├── __init__.py
│       ├── eval_retrieval.py# Đánh giá benchmark truy xuất (Hit@1, Hit@5, MRR)
│       ├── ingest.py        # Nạp và nhúng tài liệu PDF vào ChromaDB
│       └── search.py        # Tìm kiếm ngữ nghĩa trực tiếp từ terminal
├── tests/
│   ├── conftest.py          # Fixtures dùng chung cho kiểm thử (fake LLM, client)
│   ├── test_api.py          # Kiểm thử các endpoint chat, stream, extract
│   ├── test_chunker.py      # Kiểm thử thuật toán chia nhỏ văn bản
│   ├── test_config.py       # Kiểm thử quản lý biến môi trường tập trung
│   ├── test_parser.py       # Kiểm thử trích xuất và lọc header/footer PDF
│   └── test_rag.py          # Kiểm thử pipeline hỏi đáp RAG và guardrail
├── Dockerfile               # Đóng gói image container Python 3.12-slim
├── docker-compose.yml       # Điều phối dịch vụ và volume lưu trữ
├── requirements.txt         # Danh sách các thư viện phụ thuộc
├── .env.example             # Mẫu khai báo biến môi trường chuẩn
├── .dockerignore            # Loại bỏ các file rác khi build image
└── .gitignore               # Bảo vệ các file bí mật và dữ liệu runtime
```

---

## 🛠️ Yêu Cầu Hệ Thống

- **Python:** 3.11 hoặc 3.12+
- **NVIDIA API Key:** Đăng ký miễn phí tại [NVIDIA NIM](https://build.nvidia.com/)
- **Docker & Docker Compose** (Tùy chọn nếu chạy container)

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
Sao chép file `.env.example` thành `.env` và điền API key của bạn:
```ini
# Bắt buộc:
NVIDIA_API_KEY=nvapi-your-key-here

# Tùy chọn (đã có giá trị mặc định tối ưu):
LLM_MODEL=deepseek-ai/deepseek-v4.1-flash
DB_PATH=docmind.db
CHROMA_PATH=chroma_data
EMBEDDING_MODEL=BAAI/bge-M3
RAG_TOP_K=5
RAG_MIN_SCORE=0.5
CHUNK_SIZE=1500
CHUNK_OVERLAP=200
```

### 3. Khởi động ứng dụng
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Truy cập tài liệu API tương tác tại: **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 📄 Hướng Dẫn Sử Dụng RAG & Công Cụ CLI

### 1. Nạp tài liệu PDF vào hệ thống
Nạp và nhúng toàn bộ nội dung file PDF vào cơ sở dữ liệu vector:
```bash
python -m app.scripts.ingest "duong/dan/toi/tai_lieu.pdf"
```

### 2. Tìm kiếm ngữ nghĩa nhanh qua terminal
Kiểm tra kết quả truy xuất các đoạn văn bản tương đồng:
```bash
python -m app.scripts.search "khái niệm về AI Agent"
```

### 3. Đánh giá chất lượng truy xuất (Retrieval Benchmark)
Đo lường các chỉ số Hit@1, Hit@5, MRR và quét ngưỡng điểm tin cậy:
```bash
python -m app.scripts.eval_retrieval "app/eval/questions.jsonl"
```

---

## 🐳 Hướng Dẫn Chạy Bằng Docker

```bash
# Khởi chạy container ngầm
docker compose up --build -d

# Xem logs hoạt động
docker compose logs -f

# Dừng dịch vụ
docker compose down
```

---

## 🧪 Chạy Kiểm Thử (Testing)

Dự án đi kèm bộ kiểm thử tự động toàn diện, chạy hoàn toàn offline không tốn quota API nhờ hệ thống giả lập:

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
| `GET` | `/conversations/{conv_id}` | Lấy toàn bộ lịch sử tin nhắn của cuộc hội thoại |
| `POST` | `/extract` | Trích xuất thông tin cấu trúc từ văn bản tự do |
| `POST` | `/ask` | **Hỏi đáp RAG trên tài liệu**, kèm số trang và nguồn tham khảo |
| `GET` | `/documents` | **Liệt kê danh sách tài liệu** đã nạp trong vector store |
