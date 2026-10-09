
from fastembed import SparseTextEmbedding

model = SparseTextEmbedding(model_name="Qdrant/bm25")

text = "ASUS laptop with Intel Core i5, 8 GB RAM and 512 GB SSD"

embedding = next(model.embed([text]))

print("Sparse indices:", embedding.indices[:10])
print("Sparse values:", embedding.values[:10])
print("Number of nonzero dimensions:", len(embedding.indices))
