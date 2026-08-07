"""
Retrieval service (Points 1, 6, 7).

Point 7 (Performance) fix in this file:
The original code called get_chroma_client() and get_chroma_collection()
on every single request. chromadb.HttpClient() opens a fresh HTTP connection
pool each time it's constructed - there's no reason to do that per-request
when the client is stateless and thread-safe for our usage pattern. We now
build it once per process via lru_cache and reuse it.

Point 1 (Retrieval quality) - the strategy, in order of precedence:
1. Exact/substring product name match: if a known product name (or a close
   variant of one) appears in the query, we go straight to a Chroma metadata
   filter (`where={"name": ...}`) instead of relying purely on semantic
   nearest-neighbor search. This avoids the classic RAG failure mode where
   a semantically-similar-but-wrong product outranks the exact one the user
   named (e.g. "apple juice" pulling back "apple pomace" instead).
2. Fuzzy match fallback: uses difflib (stdlib, no new dependency) to catch
   misspellings ("eggfriut", "aple juice") that a substring check would miss.
   A configurable similarity threshold (default 0.72) avoids false positives
   on short/common words.
3. Semantic fallback: if no product name is confidently identified, we fall
   back to embedding-based nearest-neighbor search, but pull more candidates
   than we need (default 8) and rerank down to the final top_k (default 3),
   rather than trusting raw vector distance alone as the final ranking.

Point 6 (Reranking) - kept deliberately lightweight per the "don't
overengineer" constraint: rather than pulling in a cross-encoder model
(extra ~100-500MB dependency + another cold-start download to bake into the
image), we rerank candidates by combining Chroma's cosine distance with a
cheap keyword-overlap bonus computed against the product name/description.
This catches cases where the closest embedding isn't the best lexical match.
If retrieval quality still isn't good enough after this, the natural next
upgrade is a sentence-transformers CrossEncoder (e.g.
cross-encoder/ms-marco-MiniLM-L-6-v2) reranking the same candidate set -
the RetrievedProduct/rerank interfaces below are already shaped so that
would be a drop-in replacement for `_rerank()` without touching callers.
"""
import difflib
import re
import time
from functools import lru_cache
from typing import Optional

import chromadb
import requests

from app.config import get_settings
from app.logging_config import get_logger
from app.models import RetrievedProduct
from app.services.embedding_service import embed_texts

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[a-zA-Z]+")

# Words to exclude from keyword-overlap scoring in _rerank(). Two categories:
# 1. Catalog branding ("juice", "shop", "owasp") - appears in nearly every
#    product's name/description because of the "OWASP Juice Shop" branding,
#    so it matches almost anything and provides zero discriminative signal.
#    Without this, non-juice items (a CTF book, a t-shirt) can rank above
#    actual juice products purely because their title contains "Juice Shop."
# 2. Generic query filler words that carry no product-identifying meaning.
_KEYWORD_OVERLAP_STOPWORDS = {
    # NOTE: "juice" is deliberately NOT in this list. It's ambiguous in this
    # catalog - meaningless noise in branding items like "Pwning OWASP Juice
    # Shop," but a genuinely meaningful product-category word in real juice
    # products like "Apple Juice." Excluding it entirely broke the ability
    # to distinguish "Apple Juice" from "Apple Pomace" on an "apple juice"
    # query. "owasp" and "shop" are the actual branding-only artifact - they
    # never appear in real juice product names, only in merchandise/book titles.
    "shop", "owasp",
    "the", "a", "an", "of", "for", "is", "are", "do", "does", "you", "your",
    "what", "which", "would", "recommend", "need", "have", "has", "i", "me",
    "my", "with", "help", "please", "can", "could", "to", "it", "its", "how",
    "much", "about", "tell",
}


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.HttpClient:
    settings = get_settings()
    return chromadb.HttpClient(
        host=settings.chroma_server_host,
        port=settings.chroma_server_http_port,
    )


@lru_cache(maxsize=1)
def get_chroma_collection():
    settings = get_settings()
    client = get_chroma_client()
    return client.get_or_create_collection(name=settings.chroma_collection_name)


# In-memory cache of known product names, refreshed on ingest. Used for
# exact/fuzzy name matching without hitting Chroma just to list products.
_known_product_names: list[str] = []


def _refresh_known_product_names(names: list[str]) -> None:
    global _known_product_names
    # Longest names first so substring matching prefers more specific names
    # (e.g. "Eggfruit Juice (500ml)" over a shorter partial collision).
    _known_product_names = sorted(set(names), key=len, reverse=True)


def _find_named_product(query: str) -> Optional[str]:
    """Returns the canonical product name if the query appears to be about
    a specific known product (via substring or fuzzy match), else None."""
    if not _known_product_names:
        return None

    query_lower = query.lower()

    # 1. Substring match - cheap, precise, catches exact/partial names.
    for name in _known_product_names:
        base_name = re.sub(r"\s*\(.*?\)\s*", "", name).strip().lower()
        if base_name and base_name in query_lower:
            return name

    # 2. Fuzzy match on individual query tokens/n-grams for misspellings.
    settings = get_settings()
    tokens = _WORD_RE.findall(query_lower)
    candidates = [
        re.sub(r"\s*\(.*?\)\s*", "", name).strip().lower()
        for name in _known_product_names
    ]
    best_name, best_ratio = None, 0.0
    for n in (3, 2, 1):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            match = difflib.get_close_matches(phrase, candidates, n=1, cutoff=settings.fuzzy_match_threshold)
            if match:
                ratio = difflib.SequenceMatcher(None, phrase, match[0]).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_name = _known_product_names[candidates.index(match[0])]
    return best_name


def _keyword_overlap_score(query: str, product: RetrievedProduct) -> float:
    query_tokens = set(_WORD_RE.findall(query.lower())) - _KEYWORD_OVERLAP_STOPWORDS
    text_tokens = set(_WORD_RE.findall(f"{product.name} {product.description}".lower())) - _KEYWORD_OVERLAP_STOPWORDS
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    return overlap / max(len(query_tokens), 1)


def _rerank(query: str, candidates: list[RetrievedProduct], top_k: int) -> list[RetrievedProduct]:
    """Combines vector distance (lower = better) with keyword overlap
    (higher = better) into one score, sorts, returns top_k. See module
    docstring for why this is a lightweight rerank rather than a cross-encoder."""
    def combined_score(p: RetrievedProduct) -> float:
        distance_score = -(p.distance if p.distance is not None else 1.0)
        overlap_bonus = _keyword_overlap_score(query, p) * 0.5
        return distance_score + overlap_bonus

    ranked = sorted(candidates, key=combined_score, reverse=True)
    return ranked[:top_k]


def _row_to_product(id_: str, metadata: dict, distance: Optional[float], match_reason: str) -> RetrievedProduct:
    return RetrievedProduct(
        id=id_,
        name=metadata.get("name", ""),
        description=metadata.get("description", ""),
        price=str(metadata.get("price", "")),
        deluxe_price=str(metadata.get("deluxePrice", "")),
        image=metadata.get("image", ""),
        distance=distance,
        match_reason=match_reason,
    )


def ingest_products_to_chroma() -> int:
    settings = get_settings()
    response = requests.get(settings.product_api_url, timeout=20)
    response.raise_for_status()
    data = response.json()
    products = data.get("data", []) if isinstance(data, dict) else []
    if not isinstance(products, list):
        raise ValueError("Invalid product data from app API")

    ids, documents, metadatas = [], [], []
    for product in products:
        product_id = str(product.get("id", ""))
        if not product_id:
            continue
        ids.append(product_id)
        documents.append(
            f"Name: {product.get('name', '')}\n"
            f"Description: {product.get('description', '')}\n"
            f"Price: {product.get('price', '')}\n"
            f"Deluxe price: {product.get('deluxePrice', '')}\n"
            f"Image: {product.get('image', '')}"
        )
        metadatas.append({
            "name": product.get("name", ""),
            "description": product.get("description", ""),
            "price": product.get("price", ""),
            "deluxePrice": product.get("deluxePrice", ""),
            "image": product.get("image", ""),
        })

    collection = get_chroma_collection()
    if ids:
        try:
            collection.delete(ids=ids)
        except Exception:
            pass
        embeddings = embed_texts(documents)
        collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    _refresh_known_product_names([m["name"] for m in metadatas if m.get("name")])
    logger.info("ingest_complete", extra={"event": "ingest_complete", "count": len(ids)})
    return len(ids)


def retrieve_relevant_products(query: str) -> list[RetrievedProduct]:
    """Main retrieval entry point implementing the strategy described in the
    module docstring: named-product lookup -> semantic fallback -> rerank."""
    settings = get_settings()
    start = time.perf_counter()
    collection = get_chroma_collection()

    matched_name = _find_named_product(query)
    match_reason = "semantic"
    results_ids: list[str] = []
    results_metadatas: list[dict] = []
    results_distances: list[Optional[float]] = []

    if matched_name:
        filtered = collection.get(where={"name": matched_name})
        if filtered and filtered.get("ids"):
            results_ids = filtered["ids"]
            results_metadatas = filtered["metadatas"]
            results_distances = [None] * len(results_ids)
            match_reason = "metadata_filter"

    if not results_ids:
        query_embedding = embed_texts([query])[0]
        semantic = collection.query(
            query_embeddings=[query_embedding],
            n_results=settings.retrieval_candidate_count,
        )
        results_ids = semantic.get("ids", [[]])[0]
        results_metadatas = semantic.get("metadatas", [[]])[0]
        results_distances = semantic.get("distances", [[]])[0] if semantic.get("distances") else [None] * len(results_ids)

    candidates = [
        _row_to_product(id_, meta, dist, match_reason)
        for id_, meta, dist in zip(results_ids, results_metadatas, results_distances)
    ]

    top_products = _rerank(query, candidates, settings.retrieval_top_k) if match_reason == "semantic" else candidates[: settings.retrieval_top_k]

    # Confidence gate: exact/fuzzy name matches are confident by construction
    # (the user named a real product). Semantic-fallback matches are not -
    # nearest-neighbor search always returns *something*, even when nothing
    # is actually relevant (e.g. "I need help with my order" still returns
    # the 3 least-dissimilar products, all of which are genuinely unrelated).
    # If the best semantic match's distance is worse than our threshold,
    # treat it as no match at all rather than handing weak, misleading
    # context to the LLM.
    if match_reason == "semantic" and top_products:
        best_distance = top_products[0].distance
        if best_distance is not None and best_distance > settings.retrieval_distance_threshold:
            logger.info(
                "retrieval_low_confidence_discarded",
                extra={
                    "event": "retrieval_low_confidence_discarded",
                    "best_distance": best_distance,
                    "threshold": settings.retrieval_distance_threshold,
                },
            )
            top_products = []

    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        "retrieval_complete",
        extra={
            "event": "retrieval_complete",
            "match_reason": match_reason,
            "matched_name": matched_name,
            "candidate_count": len(candidates),
            "returned_count": len(top_products),
            "retrieval_latency_ms": elapsed_ms,
        },
    )
    return top_products
