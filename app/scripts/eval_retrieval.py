import json
import sys
from collections import defaultdict
from pathlib import Path

from app.rag import embedder, store

K = 5
path = Path(sys.argv[1] if len(sys.argv) > 1 else "eval/questions.jsonl")
items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

vectors = embedder.embed_texts([i["question"] for i in items])

def expand(pages: list[str]) -> set[str]:
    out: set[str] = set()
    for p in pages:
        if "-" in p:
            a, b = p.split("-", 1)
            out.update(str(n) for n in range(int(a), int(b) + 1))
        else:
            out.add(p)
    return out


rows = []
for item, vec in zip(items, vectors):
    hits = store.search(vec, k=K)
    labels = [h["page_label"] for h in hits]
    rank = next((r for r, lab in enumerate(labels, 1) if lab in expand(item["pages"])), None)
    rows.append({**item, "rank": rank, "top_score": hits[0]["score"] if hits else 0.0, "labels": labels, "hits": hits})

answerable = [r for r in rows if r["pages"]]
no_answer = [r for r in rows if not r["pages"]]


print(f"\n=== Chi tiết ({len(rows)} câu) ===")
for r in rows:
    if r["pages"]:
        status = f"hạng {r['rank']}" if r["rank"] else "KHÔNG TÌM THẤY"
    else:
        status = "(không có đáp án)"
    print(f"[{r['type']:10}] top1={r['top_score']:.3f} | {status:16} | {r['question']}")
    if r["pages"] and not r["rank"]:
        print(f"             cần trang {r['pages']}, nhận được {r['labels']}")
        for h in r["hits"][:3]:
            print(f"               - trang {h['page_label']} ({h['score']:.3f}): {h['text'][:100]!r}")

n = len(answerable)
print(f"\n=== Chỉ số (trên {n} câu có đáp án) ===")
print(f"Hit@1: {sum(1 for r in answerable if r['rank'] == 1)}/{n}")
print(f"Hit@{K}: {sum(1 for r in answerable if r['rank'])}/{n}")
print(f"MRR:   {sum(1 / r['rank'] for r in answerable if r['rank']) / n:.3f}")

print("\n=== Theo loại câu hỏi (Hit@5) ===")
by_type = defaultdict(list)
for r in answerable:
    by_type[r["type"]].append(r["rank"] is not None)
for t, flags in by_type.items():
    print(f"{t:10}: {sum(flags)}/{len(flags)}")

print("\n=== Quét ngưỡng điểm (so với top-1) ===")
print("ngưỡng | câu có đáp án vẫn được trả lời | câu không đáp án bị từ chối đúng")
for t in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
    kept = sum(1 for r in answerable if r["top_score"] >= t)
    refused = sum(1 for r in no_answer if r["top_score"] < t)
    print(f"{t:.2f}   | {kept}/{n}".ljust(40) + f"| {refused}/{len(no_answer)}")


meta = store.get_collection().get(include=["metadatas"])["metadatas"]
all_labels = [m["page_label"] for m in meta]
total = len(all_labels)

random_hit = 0.0
print("\n=== Độ rộng đáp án chuẩn ===")
for r in answerable:
    allowed = expand(r["pages"])
    r["p"] = sum(1 for lab in all_labels if lab in allowed) / total
    random_hit += 1 - (1 - r["p"]) ** K
    if r["p"] > 0.05:
        print(f"RỘNG ({r['p']:.0%} số chunk): {r['question']}")
print(f"Hit@{K} nếu truy xuất NGẪU NHIÊN (ước lượng): {random_hit:.1f}/{n}")

narrow = [r for r in answerable if r["p"] <= 0.05]
print(
    f"Chỉ tính {len(narrow)} câu có đáp án hẹp (<=5% số chunk): "
    f"Hit@1 = {sum(1 for r in narrow if r['rank'] == 1)}/{len(narrow)}, "
    f"Hit@{K} = {sum(1 for r in narrow if r['rank'])}/{len(narrow)}"
)