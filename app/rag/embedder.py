from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer

from app.config import settings

MODEL_NAME = settings.EMBEDDING_MODEL

@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    if device == "cuda":
        model.half()  # fp16
    print(f"[embedder] {MODEL_NAME} chạy trên {device}")
    return model


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 100
    )
    return vectors.tolist()
