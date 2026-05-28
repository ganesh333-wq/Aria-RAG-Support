"""
──────────────────────────────────────────────────────────────────────────────
 all-MiniLM-L6-v2 Embedding Model — Singleton wrapper

 Uses LangChain's HuggingFaceEmbeddings backed by sentence-transformers.
 Produces 384-dimensional normalized vectors ideal for cosine similarity.

 Why all-MiniLM-L6-v2?
   - Fast: ~14,000 sentences/sec on CPU
   - Accurate: Strong semantic similarity performance on MTEB benchmarks
   - Lightweight: 80MB model size
   - Normalized: Vectors are L2-normalized → dot product == cosine similarity
──────────────────────────────────────────────────────────────────────────────
"""

import warnings
from typing import Optional

# Suppress HuggingFace tokenizer warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_community.embeddings import HuggingFaceEmbeddings


# ── Singleton instance ────────────────────────────────────────────────────────

_embedding_instance: Optional[HuggingFaceEmbeddings] = None

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return the singleton HuggingFaceEmbeddings instance.
    Lazy-loads the model on first call (takes ~2-5 seconds).

    The model encodes text into 384-dimensional vectors normalized for cosine
    similarity. Two semantically similar sentences will have a high dot product
    (close to 1.0) even if they use different words.
    """
    global _embedding_instance
    if _embedding_instance is None:
        print(f"[Embeddings] Loading {MODEL_NAME} ...")
        _embedding_instance = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,   # L2 normalization → cosine = dot product
                "batch_size": 64,
            },
        )
        print(f"[Embeddings] Model loaded — dim={EMBEDDING_DIM}")
    return _embedding_instance
