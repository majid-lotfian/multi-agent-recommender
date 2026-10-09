
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "laptops"

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=768,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )
    print("Collection created successfully!")
else:
    print("Collection already exists!")
