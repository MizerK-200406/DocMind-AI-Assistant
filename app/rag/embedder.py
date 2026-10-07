"""Module chuyển đổi văn bản thành vector biểu diễn (Dense Text Embeddings).

Sử dụng mô hình SentenceTransformer (mặc định BAAI/bge-M3) hỗ trợ đa ngôn ngữ.
Tối ưu hóa:
- Cache singleton instance bằng @lru_cache để không tải lại trọng số mô hình nhiều lần.
- Tự động nhận diện phần cứng GPU (CUDA) và kích hoạt chế độ FP16 nửa độ chính xác để tăng tốc độ nhúng.
"""

from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer

from app.config import settings

MODEL_NAME = settings.EMBEDDING_MODEL

@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Tải và khởi tạo mô hình embedding dạng singleton có cache."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    if device == "cuda":
        model.half()  # fp16 tăng tốc xử lý và tiết kiệm VRAM
    print(f"[embedder] {MODEL_NAME} chạy trên {device}")
    return model


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Tạo vector embeddings chuẩn hóa L2 cho danh sách văn bản theo từng batch."""
    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 100
    )
    return vectors.tolist()
