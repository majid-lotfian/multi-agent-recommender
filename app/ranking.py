
from app.schemas import LaptopPreferences


def _contains(value, expected):
    return expected.lower() in str(value or "").lower()


def satisfies_constraints(product, p: LaptopPreferences):
    numeric_rules = [
        ("price", p.min_price, p.max_price),
        ("ram_gb", p.min_ram_gb, p.max_ram_gb),
        ("storage_gb", p.min_storage_gb, p.max_storage_gb),
        ("screen_inches", p.min_screen_inches, p.max_screen_inches),
    ]

    for field, minimum, maximum in numeric_rules:
        if minimum is None and maximum is None:
            continue

        value = product.get(field)
        if value is None:
            return False

        try:
            value = float(value)
        except (ValueError, TypeError):
            return False

        if minimum is not None and value < minimum:
            return False
        if maximum is not None and value > maximum:
            return False

    for field, required in [
        ("brand", p.required_brand),
        ("cpu", p.required_cpu),
        ("gpu", p.required_gpu),
    ]:
        if required and not _contains(product.get(field), required):
            return False

    if p.touchscreen_required is not None:
        is_touch = str(product.get("touch", "")).lower() in (
            "yes", "true", "1"
        )
        if is_touch != p.touchscreen_required:
            return False

    return True


def rank_products(products, preferences: LaptopPreferences):
    ranked = []

    for product in products:
        if not satisfies_constraints(product, preferences):
            continue

        score = 0.0
        reasons = []

        # Retrieval relevance is already ordered by RRF.
        # Use it as one signal, not as a hardware quality score.
        score += float(product.get("retrieval_score", 0)) * 10

        for field, preferred, weight in [
            ("brand", preferences.preferred_brand, 3.0),
            ("cpu", preferences.preferred_cpu, 3.0),
            ("gpu", preferences.preferred_gpu, 4.0),
        ]:
            if preferred and _contains(product.get(field), preferred):
                score += weight
                reasons.append(f"Matches preferred {field}")

        ram = product.get("ram_gb")
        storage = product.get("storage_gb")

        if ram is not None:
            score += min(float(ram) / 16, 2.0)

        if storage is not None:
            score += min(float(storage) / 512, 2.0)

        price = product.get("price")
        if price is not None and preferences.max_price:
            score += max(
                0,
                (preferences.max_price - float(price))
                / preferences.max_price,
            )

        ranked.append({
            **product,
            "ranking_score": round(score, 4),
            "ranking_reasons": reasons,
        })

    ranked.sort(
        key=lambda item: item["ranking_score"],
        reverse=True,
    )

    return ranked
