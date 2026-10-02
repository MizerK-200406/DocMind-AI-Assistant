# Sử dụng image Python 3.12 bản rút gọn (slim) để giảm kích thước image
FROM python:3.12-slim

# Thiết lập thư mục làm việc chính trong container
WORKDIR /code

# Cấu hình biến môi trường hệ thống:
# - PYTHONDONTWRITEBYTECODE=1: Không sinh các file bytecode .pyc thừa trong container
# - PYTHONUNBUFFERED=1: In log trực tiếp ra console mà không đệm, giúp theo dõi log dễ dàng
# - DB_PATH: Chỉ định vị trí lưu trữ file SQLite nằm trong thư mục /data (nơi gắn volume)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DB_PATH=/data/docmind.db

# Tạo trước thư mục /data để làm mount point cho persistent volume
RUN mkdir -p /data

# Sao chép và cài đặt các thư viện phụ thuộc trước để tận dụng Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn ứng dụng vào container
COPY app ./app

# Mở cổng 8000 cho ứng dụng web FastAPI
EXPOSE 8000

# Lệnh khởi chạy server Uvicorn khi container bắt đầu chạy
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
