
import os
from functools import lru_cache

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from google import genai
from google.genai import types
from qdrant_client import QdrantClient, models

load_dotenv()

COLLECTION = "laptops"


@lru_cache(maxsize=1)
def get_vertex_client():
    return genai.Client(
        vertexai=True,
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION"),
    )


@lru_cache(maxsize=1)
def get_qdrant_client():
    return QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333")
    )


@lru_cache(maxsize=1)
def get_bm25_model():
    return SparseTextEmbedding(model_name="Qdrant/bm25")


def embed_query(query: str):
    response = get_vertex_client().models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    return response.embeddings[0].values


def build_filters(
    min_price=None,
    max_price=None,
    min_ram_gb=None,
    max_ram_gb=None,
    min_storage_gb=None,
    max_storage_gb=None,
    min_screen_inches=None,
    max_screen_inches=None,
    brand=None,
):
    conditions = []

    ranges = [
        ("price", min_price, max_price),
        ("ram_gb", min_ram_gb, max_ram_gb),
        ("storage_gb", min_storage_gb, max_storage_gb),
        ("screen_inches", min_screen_inches, max_screen_inches),
    ]

    for field, minimum, maximum in ranges:
        if minimum is not None or maximum is not None:
            conditions.append(
                models.FieldCondition(
                    key=field,
                    range=models.Range(
                        gte=minimum,
                        lte=maximum,
                    ),
                )
            )

    if brand:
        conditions.append(
            models.FieldCondition(
                key="brand",
                match=models.MatchValue(value=brand),
            )
        )

    return models.Filter(must=conditions) if conditions else None


def hybrid_search(
    query: str,
    limit: int = 50,
    **filters,
):
    dense = embed_query(query)

    sparse = next(
        get_bm25_model().query_embed(query)
    )

    sparse_vector = models.SparseVector(
        indices=sparse.indices.tolist(),
        values=sparse.values.tolist(),
    )

    query_filter = build_filters(**filters)

    results = get_qdrant_client().query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(
                query=dense,
                using="dense",
                filter=query_filter,
                limit=100,
            ),
            models.Prefetch(
                query=sparse_vector,
                using="bm25",
                filter=query_filter,
                limit=100,
            ),
        ],
        query=models.FusionQuery(
            fusion=models.Fusion.RRF
        ),
        limit=limit,
        with_payload=True,
    )

    return [
        {
            **(point.payload or {}),
            "id": point.id,
            "retrieval_score": point.score,
        }
        for point in results.points
    ]
