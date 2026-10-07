from app.ingest.parser import find_running_lines, strip_running_lines


def test_running_header_removed_but_content_kept():
    header = "1.1 Agent hiện đại = LLM + context + tools"
    raws = [
        f"{header}\nNội dung riêng của trang {i}.\nÝ thứ hai của trang {i}.\nÝ thứ ba của trang {i}."
        for i in range(5)
    ]
    running = find_running_lines(raws)
    assert header in running

    cleaned = strip_running_lines(raws[0], running)
    assert header not in cleaned
    assert "Nội dung riêng của trang 0." in cleaned