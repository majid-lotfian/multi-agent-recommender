
import json

from app.retrieval import hybrid_search, get_qdrant_client
from app.ranking import rank_products
from app.schemas import LaptopPreferences


def search_and_rank_laptops(
    preferences_json: str,
    limit: int = 5,
) -> dict:
    """Search and rank laptops using structured user preferences.

    Args:
        preferences_json: JSON object containing laptop preferences.
        limit: Maximum number of final recommendations.

    Returns:
        Matching laptop products with ranking scores.
    """
    preferences = LaptopPreferences.model_validate_json(
        preferences_json
    )

    search_text = " ".join(
        str(value)
        for value in [
            preferences.intended_use,
            preferences.preferred_cpu,
            preferences.preferred_gpu,
            preferences.required_cpu,
            preferences.required_gpu,
            *preferences.other_preferences,
        ]
        if value
    )

    products = hybrid_search(
        query=f"Laptop for {search_text}",
        limit=100,
        min_price=preferences.min_price,
        max_price=preferences.max_price,
        min_ram_gb=preferences.min_ram_gb,
        max_ram_gb=preferences.max_ram_gb,
        min_storage_gb=preferences.min_storage_gb,
        max_storage_gb=preferences.max_storage_gb,
        min_screen_inches=preferences.min_screen_inches,
        max_screen_inches=preferences.max_screen_inches,
        brand=preferences.required_brand,
    )

    ranked = rank_products(products, preferences)

    return {
        "count": len(ranked),
        "products": ranked[:limit],
        "message": (
            "Matching products found."
            if ranked else
            "No matching products found in the retrieved candidates."
        ),
    }


def compare_laptops(product_ids: list[int]) -> dict:
    """Retrieve and compare laptops already known by their catalog IDs.

    Args:
        product_ids: List of laptop product IDs to compare.

    Returns:
        Product specifications for a side-by-side comparison.
    """
    if not product_ids or len(product_ids) > 5:
        return {
            "error": "Provide between 1 and 5 product IDs."
        }

    points = get_qdrant_client().retrieve(
        collection_name="laptops",
        ids=product_ids,
        with_payload=True,
    )

    products = [
        {"id": point.id, **(point.payload or {})}
        for point in points
    ]

    return {
        "products": products,
        "found": len(products),
        "requested": len(product_ids),
    }
