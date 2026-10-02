import json

import httpx
from openai import APIConnectionError

from app import main

VALID_DOC = json.dumps(
    {
        "title": "Thư xin gia hạn",
        "doc_type": "email",
        "language": "vi",
        "summary": "Xin gia hạn thêm 10 ngày.",
        "key_points": ["Dự án hoàn thành 90%"],
        "sentiment": "trung_tinh",
        "confidence": 0.9,
    },
    ensure_ascii=False,
)


def test_chat_saves_conversation(client, fake_llm):
    r = client.post("/chat", json={"message": "Xin chào"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Xin chào!"

    history = client.get(f"/conversations/{body['conversation_id']}").json()["messages"]
    assert [m["role"] for m in history] == ["user", "assistant"]


def test_history_is_sent_to_llm(client, fake_llm):
    first = client.post("/chat", json={"message": "Tôi tên là Nam"}).json()
    client.post("/chat", json={"message": "Tên tôi là gì?", "conversation_id": first["conversation_id"]})
    # system + user1 + assistant1 + user2
    assert len(fake_llm.calls[1]) == 4


def test_max_history_limits_context(client, fake_llm, monkeypatch):
    monkeypatch.setattr(main, "MAX_HISTORY", 2)
    cid = client.post("/chat", json={"message": "1"}).json()["conversation_id"]
    client.post("/chat", json={"message": "2", "conversation_id": cid})
    client.post("/chat", json={"message": "3", "conversation_id": cid})
    # system + 1 tin lịch sử + tin mới
    assert len(fake_llm.calls[2]) == 3


def test_unknown_conversation_returns_404(client, fake_llm):
    r = client.post("/chat", json={"message": "hi", "conversation_id": "khong-ton-tai"})
    assert r.status_code == 404


def test_message_too_long_returns_422(client, fake_llm):
    r = client.post("/chat", json={"message": "a" * 5000})
    assert r.status_code == 422
    assert fake_llm.calls == []  # không tốn lượt gọi LLM nào


def test_llm_failure_does_not_pollute_history(client, fake_llm):
    cid = client.post("/chat", json={"message": "Xin chào"}).json()["conversation_id"]

    fake_llm.error = APIConnectionError(request=httpx.Request("POST", "http://test"))
    r = client.post("/chat", json={"message": "Câu bị lỗi", "conversation_id": cid})
    assert r.status_code == 502

    history = client.get(f"/conversations/{cid}").json()["messages"]
    assert len(history) == 2  # vẫn chỉ có cặp đầu tiên


def test_stream_saves_full_answer(client, fake_llm):
    r = client.post("/chat/stream", json={"message": "Xin chào"})
    assert "[DONE]" in r.text
    first_event = r.text.split("\n\n")[0].removeprefix("data: ")
    cid = json.loads(first_event)["conversation_id"]

    history = client.get(f"/conversations/{cid}").json()["messages"]
    assert history[-1] == {"role": "assistant", "content": "Xin chào!"}


def test_extract_valid(client, fake_llm):
    fake_llm.replies = [VALID_DOC]
    r = client.post("/extract", json={"text": "Một email xin gia hạn."})
    assert r.status_code == 200
    assert r.json()["doc_type"] == "email"


def test_extract_retries_on_invalid_json(client, fake_llm):
    fake_llm.replies = ["đây không phải JSON", VALID_DOC]
    r = client.post("/extract", json={"text": "Một email xin gia hạn."})
    assert r.status_code == 200
    assert len(fake_llm.calls) == 2
    assert "JSON chưa hợp lệ" in fake_llm.calls[1][-1]["content"]  # lỗi được gửi lại cho mô hình