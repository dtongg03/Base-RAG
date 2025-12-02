from sentence_transformers import SentenceTransformer
import torch
import numpy as np


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = SentenceTransformer(
    "dangvantuan/vietnamese-document-embedding",
    trust_remote_code=True,
).to(DEVICE)


def embed_texts(texts, batch_size: int = 32) -> np.ndarray:
    """
    Nhận list[str] -> trả về np.ndarray shape (n_samples, dim)
    """
    embeddings = MODEL.encode(
        texts,
        batch_size=batch_size,
        device=DEVICE,
        show_progress_bar=True,
    )
    return np.array(embeddings)

def load_model(texts):
    return embed_texts(texts)
