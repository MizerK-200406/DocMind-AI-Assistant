from collections import Counter
from dataclasses import dataclass
import re
import unicodedata

from pypdf import PdfReader

DOT_LEADER = re.compile(r"(?:\s?\.){4,}")                 # ". . . . ." hoặc "....."
PAGE_NUMBER_LINE = re.compile(r"(?m)^[ \t]*\d{1,4}[ \t]*$")  # dòng chỉ có số
TRAILING_PAGE_NO = re.compile(r"[ \t]+\d{1,4}$")


@dataclass
class Page:
    page_number: int
    label: str
    text: str


def is_toc_page(raw: str) -> bool:
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        return False
    leader_lines = sum(1 for l in lines if DOT_LEADER.search(l))
    return leader_lines / len(lines) > 0.2


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00ad", "")            # soft hyphen
    text = DOT_LEADER.sub(" ", text)
    text = PAGE_NUMBER_LINE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _norm(line: str) -> str:
    return TRAILING_PAGE_NO.sub("", line.strip())  # "Chương 1   37" và "Chương 1   38" coi như một


def _edge_indices(lines: list[str]) -> set[int]:
    idx = [i for i, l in enumerate(lines) if l.strip()]
    return set(idx[:2] + idx[-2:])


def find_running_lines(raw_pages: list[str], min_pages: int = 3) -> dict[str, int]:
    counter: Counter = Counter()
    for raw in raw_pages:
        lines = raw.splitlines()
        counter.update({_norm(lines[i]) for i in _edge_indices(lines)})  # mỗi trang tính một lần
    return {l: c for l, c in counter.items() if c >= min_pages and 6 <= len(l) <= 120}


def strip_running_lines(raw: str, running: dict[str, int]) -> str:
    lines = raw.splitlines()
    edges = _edge_indices(lines)
    kept = [l for i, l in enumerate(lines) if not (i in edges and _norm(l) in running)]
    return "\n".join(kept)


def parse_pdf(path: str, verbose: bool = False) -> list[Page]:
    reader = PdfReader(path)
    labels = reader.page_labels
    raws = [(page.extract_text() or "") for page in reader.pages]

    running = find_running_lines([r for r in raws if not is_toc_page(r)])
    if verbose:
        print(f"Phát hiện {len(running)} dòng lặp ở mép trang. Nhiều nhất:")
        for line, count in sorted(running.items(), key=lambda x: -x[1])[:20]:
            print(f"  {count:3} trang | {line}")

    pages: list[Page] = []
    for i, raw in enumerate(raws, start=1):
        if is_toc_page(raw):
            continue
        text = clean_text(strip_running_lines(raw, running))
        if text:
            pages.append(Page(page_number=i, label=labels[i - 1], text=text))
    return pages
