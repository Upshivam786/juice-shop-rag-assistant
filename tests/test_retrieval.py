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
