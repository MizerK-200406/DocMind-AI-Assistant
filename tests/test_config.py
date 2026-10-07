import os
from unittest.mock import patch


def test_default_settings():
    from app.config import Settings

    with patch.dict(os.environ, {}, clear=True):
        s = Settings()
        assert s.DB_PATH == "docmind.db"
        assert s.LLM_MODEL == "deepseek-ai/deepseek-v4.1-flash"
        assert s.LLM_BASE_URL == "https://integrate.api.nvidia.com/v1"
        assert s.LLM_TIMEOUT == 120.0
        assert s.LLM_MAX_OUTPUT_TOKENS == 2048
        assert s.CHROMA_PATH == "chroma_data"
        assert s.CHROMA_COLLECTION == "docmind_chunks"
        assert s.EMBEDDING_MODEL == "BAAI/bge-M3"
        assert s.RAG_TOP_K == 5
        assert s.RAG_MIN_SCORE == 0.5
        assert s.CHUNK_SIZE == 1500
        assert s.CHUNK_OVERLAP == 200
        assert s.MAX_HISTORY == 20
        assert "Bạn là trợ lý AI" in s.SYSTEM_PROMPT


def test_custom_env_override():
    from app.config import Settings

    custom_env = {
        "DB_PATH": "custom.db",
        "LLM_MODEL": "custom-model",
        "RAG_TOP_K": "10",
        "RAG_MIN_SCORE": "0.75",
        "CHUNK_SIZE": "800",
        "CHUNK_OVERLAP": "100",
    }
    with patch.dict(os.environ, custom_env, clear=True):
        s = Settings()
        assert s.DB_PATH == "custom.db"
        assert s.LLM_MODEL == "custom-model"
        assert s.RAG_TOP_K == 10
        assert s.RAG_MIN_SCORE == 0.75
        assert s.CHUNK_SIZE == 800
        assert s.CHUNK_OVERLAP == 100


def test_singleton_settings_instance():
    from app.config import settings

    assert settings is not None
    assert hasattr(settings, "DB_PATH")
