import os

os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:9999/v1")
os.environ.setdefault("LLM_MODEL", "test-model")

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import db, llm, main


def make_response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=None,
    )


def make_chunk(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def fake_llm(monkeypatch):
    state = SimpleNamespace(calls=[], replies=[], default="Xin chào!", error=None)

    def fake(messages, **kwargs):
        state.calls.append(list(messages))
        if state.error:
            raise state.error
        if kwargs.get("stream"):
            return iter([make_chunk("Xin "), make_chunk("chào!")])
        reply = state.replies.pop(0) if state.replies else state.default
        return make_response(reply)

    monkeypatch.setattr(llm, "chat_completion", fake)
    return state