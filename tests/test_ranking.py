
import pandas as pd
import pytest

from app.schemas import LaptopPreferences
from app.ranking import (
    satisfies_constraints,
    rank_products,
)


@pytest.fixture(scope="module")
def products():
    df = pd.read_csv("data/raw/laptops.csv")

    return [
        {
            "id": int(row["id"]),
            "name": row["Laptop"],
            "brand": row["Brand"],
            "cpu": row["CPU"],
            "gpu": (
                None if pd.isna(row["GPU"])
                else row["GPU"]
            ),
            "ram_gb": int(row["RAM"]),
            "storage_gb": int(row["Storage"]),
            "screen_inches": (
                None if pd.isna(row["Screen"])
                else float(row["Screen"])
            ),
            "price": float(row["Final Price"]),
            "touch": row["Touch"],
            "retrieval_score": 0.01,
        }
        for _, row in df.iterrows()
    ]


@pytest.mark.parametrize(
    "preferences",
    [
        LaptopPreferences(max_price=1000),
        LaptopPreferences(min_price=1000, max_price=1500),
        LaptopPreferences(min_ram_gb=32),
        LaptopPreferences(min_ram_gb=16, max_price=1200),
        LaptopPreferences(required_brand="Lenovo"),
        LaptopPreferences(min_storage_gb=1000),
        LaptopPreferences(required_gpu="RTX 4060"),
    ],
)
def test_hard_constraints(products, preferences):
    matches = rank_products(products, preferences)

    assert matches, "No products matched the test case"

    for product in matches:
        assert satisfies_constraints(product, preferences)


def test_unknown_gpu_rejected(products):
    preferences = LaptopPreferences(
        required_gpu="RTX 4060"
    )

    matches = rank_products(products, preferences)

    assert all(
        product["gpu"] is not None
        for product in matches
    )


def test_no_matching_products(products):
    preferences = LaptopPreferences(
        max_price=100,
        min_ram_gb=128,
    )

    matches = rank_products(products, preferences)

    assert matches == []


def test_ranking_order(products):
    preferences = LaptopPreferences(
        max_price=1500,
        min_ram_gb=16,
    )

    matches = rank_products(products, preferences)

    scores = [
        product["ranking_score"]
        for product in matches
    ]

    assert scores == sorted(scores, reverse=True)


def test_product_ids_unique(products):
    ids = [product["id"] for product in products]

    assert len(ids) == len(set(ids))
