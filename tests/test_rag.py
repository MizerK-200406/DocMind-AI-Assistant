from app.rag import embedder, store


def fake_hit(score, page_label="12", text="Agent là hệ thống dùng LLM để ra quyết định."):
    return {"text": text, "source": "sach.pdf", "page": 20,
            "page_label": page_label, "score": score}


def patch_retrieval(monkeypatch, hits):
    monkeypatch.setattr(embedder, "embed_texts", lambda texts, **kw: [[0.0]])
    monkeypatch.setattr(store, "search", lambda *a, **kw: hits)


def test_refuses_without_calling_llm_when_scores_low(client, fake_llm, monkeypatch):
    patch_retrieval(monkeypatch, [fake_hit(0.3)])
    body = client.post("/ask", json={"question": "Tôi là ai?"}).json()
    assert body["answered"] is False
    assert body["sources"] == []
    assert fake_llm.calls == []  # không tốn lượt gọi LLM


def test_answer_has_sources_and_context(client, fake_llm, monkeypatch):
    patch_retrieval(monkeypatch, [fake_hit(0.7), fake_hit(0.6, "13", "Đoạn hai."), fake_hit(0.4)])
    fake_llm.replies = ["Agent là hệ thống dùng LLM [1]."]
    body = client.post("/ask", json={"question": "Agent là gì?"}).json()

    assert body["answered"] is True
    assert len(body["sources"]) == 2  # đoạn 0.4 bị loại bởi ngưỡng
    prompt = fake_llm.calls[0][-1]["content"]
    assert "[1] (trang 12)" in prompt and "[2] (trang 13)" in prompt

def test_unknown_source_returns_404(client, fake_llm, monkeypatch):
    monkeypatch.setattr(store, "list_sources", lambda: ["sach.pdf"])
    r = client.post("/ask", json={"question": "Agent là gì?", "source": "string"})
    assert r.status_code == 404


def test_list_documents_returns_sources(client, monkeypatch):
    monkeypatch.setattr(store, "list_sources", lambda: ["sach1.pdf", "sach2.pdf"])
    r = client.get("/documents")
    assert r.status_code == 200
    assert r.json() == {"documents": ["sach1.pdf", "sach2.pdf"]}