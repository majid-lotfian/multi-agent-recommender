
import json
import os
from pathlib import Path
import hashlib
import time
from google.genai import errors
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from google import genai
from google.genai import types
from qdrant_client import QdrantClient, models

load_dotenv()

COLLECTION = "laptops"
LIMIT = 2160
CACHE_PATH = Path("data/processed/embedding_cache.json")

with open("data/processed/laptops.json", encoding="utf-8") as f:
    products = json.load(f)

vertex = genai.Client(
    vertexai=True,
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION"),
)

qdrant = QdrantClient(url="http://localhost:6333")
bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")


def get_cache_key(text):
    config = f"gemini-embedding-001:768:RETRIEVAL_DOCUMENT:{text}"
    return hashlib.sha256(config.encode("utf-8")).hexdigest()


def generate_embedding(text, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = vertex.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=768,
                ),
            )
            return response.embeddings[0].values

        except errors.APIError as e:
            if e.code not in (429, 500, 502, 503, 504):
                raise

            if attempt == max_retries - 1:
                raise

            wait_seconds = min(2 ** (attempt + 1), 30)
            print(f"API error {e.code}. Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)

# Load cached dense embeddings
if CACHE_PATH.exists():
    with open(CACHE_PATH, encoding="utf-8") as f:
        cache = json.load(f)
else:
    cache = {}

for product in products[:LIMIT]:
    product_id = str(product["id"])
    text = product["embedding_text"]
    cache_key = get_cache_key(text)

    # Dense embedding from Vertex AI
    if cache_key not in cache:
        cache[cache_key] = generate_embedding(text)

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f)

    dense_vector = cache[cache_key]

    # Sparse embedding using local BM25
    sparse = next(bm25_model.embed([text]))

    # Store both vectors and product metadata
    point = models.PointStruct(
        id=product["id"],
        vector={
            "dense": dense_vector,
            "bm25": models.SparseVector(
                indices=sparse.indices.tolist(),
                values=sparse.values.tolist(),
            ),
        },
        payload={
            k: v for k, v in product.items()
            if k != "embedding_text"
        },
    )

    qdrant.upsert(
        collection_name=COLLECTION,
        points=[point],
        wait=True,
    )

    print(f"Indexed product {product_id}")

print("Indexing completed!")
