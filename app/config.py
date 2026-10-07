"""Module cấu hình tập trung cho toàn bộ ứng dụng DocMind.

Gom toàn bộ các biến môi trường và thiết lập mặc định của hệ thống:
- LLM / NIM API
- SQLite Database
- ChromaDB Vector Store & Embedding Model
- RAG Pipeline parameters
- Document Ingestion chunking parameters
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """Tập trung toàn bộ thiết lập cấu hình của DocMind."""

    # LLM Settings
    NVIDIA_API_KEY: str | None = None
    LLM_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    LLM_MODEL: str = "deepseek-ai/deepseek-v4.1-flash"
    LLM_TIMEOUT: float = 120.0
    LLM_MAX_OUTPUT_TOKENS: int = 2048

    # Database Settings
    DB_PATH: str = "docmind.db"

    # Vector Store & Embedding
    CHROMA_PATH: str = "chroma_data"
    CHROMA_COLLECTION: str = "docmind_chunks"
    EMBEDDING_MODEL: str = "BAAI/bge-M3"

    # RAG Pipeline Settings
    RAG_TOP_K: int = 5
    RAG_MIN_SCORE: float = 0.5

    # Chat & API Settings
    MAX_HISTORY: int = 20
    SYSTEM_PROMPT: str = "Bạn là trợ lý AI, trả lời bằng tiếng Việt, ngắn gọn và chính xác."

    # Document Ingestion Settings
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 200

    def __init__(self):
        self.NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
        self.LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-ai/deepseek-v4.1-flash")
        self.LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
        self.LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2048"))

        self.DB_PATH = os.getenv("DB_PATH", "docmind.db")

        self.CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_data")
        self.CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "docmind_chunks")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-M3")

        self.RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
        self.RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.5"))

        self.MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))
        self.SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "Bạn là trợ lý AI, trả lời bằng tiếng Việt, ngắn gọn và chính xác.")

        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


settings = Settings()
