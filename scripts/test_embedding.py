
import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Load one laptop
with open("data/processed/laptops.json", encoding="utf-8") as f:
    products = json.load(f)

product = products[0]

# Connect to Vertex AI
client = genai.Client(
    vertexai=True,
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION"),
)

# Generate embedding
response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=product["embedding_text"],
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=768,
    ),
)

embedding = response.embeddings[0].values

print("Product:", product["name"])
print("Embedding dimensions:", len(embedding))
print("First 5 values:", embedding[:5])
