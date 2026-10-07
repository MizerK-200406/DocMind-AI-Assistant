"""Module giao tiếp với mô hình ngôn ngữ lớn (LLM).

Sử dụng thư viện OpenAI SDK để kết nối tới dịch vụ NVIDIA NIM API (mô hình DeepSeek-v4.1-flash).
Tích hợp cơ chế tự động thử lại (Retry) thông minh bằng thư viện `tenacity` đối với các lỗi tạm thời
(Transient errors) như nghẽn mạng, timeout hoặc rate-limit.
"""

import logging

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

from app.config import settings

logger = logging.getLogger("docmind.llm")

MODEL = settings.LLM_MODEL
TIMEOUT_SECONDS = settings.LLM_TIMEOUT
MAX_OUTPUT_TOKENS = settings.LLM_MAX_OUTPUT_TOKENS

# Khởi tạo client tới NVIDIA NIM. Đặt max_retries=0 để nhường toàn quyền retry cho tenacity.
client = OpenAI(
    api_key=settings.NVIDIA_API_KEY,
    base_url=settings.LLM_BASE_URL,
    timeout=TIMEOUT_SECONDS,
    max_retries=0,
)

# Các lỗi tạm thời (Transient Errors) có khả năng phục hồi khi áp dụng exponential backoff
TRANSIENT_ERROR = (
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    RateLimitError,
)


@retry(
    retry=retry_if_exception_type(TRANSIENT_ERROR),
    wait=wait_random_exponential(multiplier=1, max=20),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
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
