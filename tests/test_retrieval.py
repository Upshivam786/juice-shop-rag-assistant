"""
Retrieval tests (Point 14). These test the pure logic functions in
chroma_service.py that don't require a live Chroma server - _find_named_product
and _rerank - by directly manipulating the module's in-memory name cache and
constructing RetrievedProduct fixtures. No network/mocking needed for these.

For the parts that DO hit Chroma (retrieve_relevant_products), see
test_integration.py, which mocks the collection object instead.
"""
from app.models import RetrievedProduct
from app.services import chroma_service


def _product(name: str, description: str = "", distance: float | None = 0.2) -> RetrievedProduct:
    return RetrievedProduct(
        id=name, name=name, description=description, price="1.99",
        deluxe_price="1.99", image="x.png", distance=distance,
    )


def test_exact_substring_match_found():
    chroma_service._refresh_known_product_names([
        "Apple Juice (1000ml)", "Eggfruit Juice (500ml)", "Carrot Juice (1000ml)",
    ])
    assert chroma_service._find_named_product("how much is the apple juice") == "Apple Juice (1000ml)"


def test_fuzzy_match_handles_misspelling():
    chroma_service._refresh_known_product_names([
        "Eggfruit Juice (500ml)", "Apple Juice (1000ml)",
    ])
    # "eggfriut" is a plausible misspelling of "eggfruit"
    result = chroma_service._find_named_product("tell me about eggfriut juice")
    assert result == "Eggfruit Juice (500ml)"


def test_no_match_for_unrelated_query():
    chroma_service._refresh_known_product_names(["Apple Juice (1000ml)"])
    assert chroma_service._find_named_product("what is the capital of France") is None


def test_rerank_prefers_keyword_overlap_over_pure_distance():
    # Apple Pomace has a slightly better (lower) vector distance, but the
    # query explicitly says "apple juice" - keyword overlap should let
    # Apple Juice win the rerank despite the worse raw distance.
    apple_pomace = _product("Apple Pomace", "leftover pulp", distance=0.10)
    apple_juice = _product("Apple Juice (1000ml)", "the all-time classic apple juice", distance=0.15)

    ranked = chroma_service._rerank("apple juice", [apple_pomace, apple_juice], top_k=1)
    assert ranked[0].name == "Apple Juice (1000ml)"


def test_rerank_respects_top_k():
    products = [_product(f"Product {i}", distance=0.1 * i) for i in range(5)]
    ranked = chroma_service._rerank("juice", products, top_k=3)
    assert len(ranked) == 3


def test_rerank_ignores_catalog_branding_words():
    """Regression test: 'shop'/'owasp' appear in nearly every merchandise
    item's branding ("Pwning OWASP Juice Shop"), so they must not count
    toward keyword-overlap scoring - otherwise unrelated merchandise can
    outrank actual juice products purely because its name contains
    "OWASP Juice Shop." Distances here mirror real observed values: the
    book's raw semantic distance is worse than the juice product's, matching
    what we verified against the live catalog."""
    book = _product("Pwning OWASP Juice Shop", "companion guide to security challenges", distance=1.08)
    apple = _product("Apple Juice (1000ml)", "the all-time classic apple juice", distance=1.06)

    ranked = chroma_service._rerank("what juice would you recommend for a summer party", [book, apple], top_k=2)
    assert ranked[0].name == "Apple Juice (1000ml)"


def test_low_confidence_semantic_matches_are_discarded():
    """Regression test: when nothing in the catalog is actually relevant
    (e.g. an out-of-scope question like an order-status query), retrieval
    should recognize the weak match confidence and return nothing, rather
    than handing the LLM irrelevant products dressed up as real context."""
    from unittest.mock import patch, MagicMock
    from app.config import get_settings

    get_settings.cache_clear()

    fake_collection = MagicMock()
    fake_collection.get.return_value = {"ids": []}
    fake_collection.query.return_value = {
        "ids": [["1", "2", "3"]],
        "metadatas": [[
            {"name": "Iron-Ons", "description": "patches", "price": "14.99", "deluxePrice": "14.99", "image": "x.png"},
            {"name": "Facemask", "description": "mask", "price": "13.49", "deluxePrice": "13.49", "image": "x.png"},
            {"name": "Eggfruit Juice", "description": "exotic", "price": "8.99", "deluxePrice": "8.99", "image": "x.png"},
        ]],
        "distances": [[1.47, 1.50, 1.49]],
    }

    with patch("app.services.chroma_service.get_chroma_collection", return_value=fake_collection), \
         patch("app.services.chroma_service.embed_texts", return_value=[[0.0] * 384]):
        chroma_service._refresh_known_product_names([])
        results = chroma_service.retrieve_relevant_products("I need help with my order")

    assert results == []
