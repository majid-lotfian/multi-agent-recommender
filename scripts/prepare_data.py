
import json
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/raw/laptops.csv")
OUTPUT_PATH = Path("data/processed/laptops.json")


def clean_value(value):
    """Convert missing values into None."""
    if pd.isna(value):
        return None
    return value


def create_product_text(row):
    """Create a readable description for embedding."""

    parts = [
        str(row["Laptop"]),
        f'Brand: {row["Brand"]}',
        f'CPU: {row["CPU"]}',
        f'RAM: {row["RAM"]} GB',
        f'Storage: {row["Storage"]} GB {row["Storage type"]}',
        f'Screen: {row["Screen"]} inches',
        f'Touchscreen: {row["Touch"]}',
    ]

    if pd.notna(row["GPU"]):
        parts.append(f'GPU: {row["GPU"]}')

    return ". ".join(parts)


def prepare_data():
    df = pd.read_csv(INPUT_PATH)

    products = []

    for _, row in df.iterrows():
        product = {
            "id": int(row["id"]),
            "name": clean_value(row["Laptop"]),
            "brand": clean_value(row["Brand"]),
            "model": clean_value(row["Model"]),
            "status": clean_value(row["Status"]),
            "cpu": clean_value(row["CPU"]),
            "ram_gb": clean_value(row["RAM"]),
            "storage_gb": clean_value(row["Storage"]),
            "storage_type": clean_value(row["Storage type"]),
            "gpu": clean_value(row["GPU"]),
            "screen_inches": clean_value(row["Screen"]),
            "touch": clean_value(row["Touch"]),
            "price": clean_value(row["Final Price"]),
            "embedding_text": create_product_text(row),
        }

        products.append(product)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(products)} products")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    prepare_data()
