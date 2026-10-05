from app.ingest.chunker import split_text

SENTENCE = "Đây là một câu tiếng Việt khá dài để kiểm thử. "


def test_chunks_respect_size():
    chunks = split_text(SENTENCE * 100, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_consecutive_chunks_overlap():
    chunks = split_text(SENTENCE * 100, chunk_size=300, overlap=100)
    assert chunks[1][:20] in chunks[0]


def test_empty_text_returns_no_chunks():
    assert split_text("") == []