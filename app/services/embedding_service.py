"""
Embedding generation. Functionally identical to the embedding.py you already
have working (baked-in sentence-transformers model, no runtime downloads) -
relocated under app/services/ purely for the project structure in Point 13.
No behavior change here; this is the file your existing Dockerfile's
`RUN python -c "from sentence_transformers import SentenceTransformer; ..."`
bake-in step should continue to warm.
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    return model.encode(texts, normalize_embeddings=True).tolist()
