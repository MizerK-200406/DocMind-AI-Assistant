"""Module giao tiếp với mô hình ngôn ngữ lớn (LLM).

Sử dụng thư viện OpenAI SDK để kết nối tới dịch vụ NVIDIA NIM API (mô hình DeepSeek-v4.1-flash).
Tích hợp cơ chế tự động thử lại (Retry) thông minh bằng thư viện `tenacity` đối với các lỗi tạm thời
(Transient errors) như nghẽn mạng, timeout hoặc rate-limit.
"""

import logging
import os

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

# Nạp các biến môi trường từ file .env nếu có
load_dotenv()

# Logger chuyên biệt cho module LLM
logger = logging.getLogger("docmind.llm")

# Tên mô hình LLM sử dụng từ NVIDIA NIM catalog
MODEL = "deepseek-ai/deepseek-v4.1-flash"
TIMEOUT_SECONDS = float(30)
MAX_OUTPUT_TOKENS = 2048

# Khởi tạo client OpenAI kết nối tới endpoint của NVIDIA NIM
# Đặt max_retries=0 vì chúng ta tự quản lý retry bằng thư viện tenacity bên dưới
client = OpenAI(
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
    timeout=TIMEOUT_SECONDS,
    max_retries=0,
)

# Danh sách các ngoại lệ tạm thời (Transient Errors) có khả năng tự phục hồi khi thử lại
TRANSIENT_ERROR = (
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    RateLimitError,
)


@retry(
    # Chỉ retry khi gặp các lỗi mạng hoặc giới hạn tải tạm thời
    retry=retry_if_exception_type(TRANSIENT_ERROR),
    # Chiến lược chờ tăng theo cấp số nhân ngẫu nhiên (exponential backoff) từ 1s đến tối đa 20s
    wait=wait_random_exponential(multiplier=1, max=20),
    # Dừng sau tối đa 4 lần thử
    stop=stop_after_attempt(4),
    # Ghi log cảnh báo trước mỗi lần ngủ và thử lại
    before_sleep=before_sleep_log(logger, logging.WARNING),
    # Nếu hết số lần retry mà vẫn lỗi thì ném lại ngoại lệ gốc ra ngoài
    reraise=True,
)
def chat_completion(messages: list[dict], **kwargs):
    """Gửi yêu cầu hoàn thành đoạn chat tới mô hình LLM với cơ chế bảo vệ retry tự động.
    
    Args:
        messages: Danh sách các tin nhắn ngữ cảnh [{'role': '...', 'content': '...'}].
        **kwargs: Các tham số bổ sung như temperature, stream, max_tokens,...
        
    Returns:
        ChatCompletion | Stream: Đối tượng phản hồi từ OpenAI client.
    """
    kwargs.setdefault("max_tokens", MAX_OUTPUT_TOKENS)
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        **kwargs,
    )
