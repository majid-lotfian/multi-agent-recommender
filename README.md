# Multi-Agent Laptop Recommendation System

A conversational laptop recommendation prototype built with **Google Agent Development Kit (ADK)**, **Gemini**, **Vertex AI**, and **Qdrant**. It combines specialized agents with hybrid product retrieval, structured preference management, constraint-aware ranking, and natural-language product comparisons.

> **Project scope:** This is a technical demonstration using a static laptop catalog, not a live shopping service. Product prices and availability are not verified in real time, and recommendation relevance has not been validated against a labeled benchmark.

## Features

- **Multi-agent orchestration:** A coordinator routes requests to recommendation or comparison workflows.
- **Structured preference extraction:** A preference agent turns natural-language requirements into a validated Pydantic model.
- **Multi-turn refinement:** Follow-up requests update preferences while preserving unchanged requirements.
- **Hybrid retrieval:** Qdrant combines Gemini dense embeddings and BM25 sparse retrieval using Reciprocal Rank Fusion (RRF).
- **Constraint-aware ranking:** Mandatory requirements are enforced before eligible products are ranked using retrieval scores and soft preferences.
- **Product comparison:** A separate agent retrieves catalog items by ID and explains their differences.
- **Unit tests:** Deterministic tests cover ranking and hard-constraint behavior.

## Architecture

```text
                         User
                          |
                          v
                 Coordinator Agent
                    /          \
                   /            \
       Recommendation         Comparison
          Workflow               Agent
             |                     |
             v                     v
       Preference Agent       compare_laptops
             |                     |
             v                     v
         Search Agent        Catalog lookup
             |
             v
      search_and_rank_laptops
             |
             v
       Hybrid Retrieval
       /              \
  Gemini dense       BM25 sparse
       \              /
        \            /
      Reciprocal Rank Fusion
             |
             v
    Hard-constraint filtering
             |
             v
    Soft-preference ranking
             |
             v
     Recommendation Agent
             |
             v
      Explained suggestions
```

### Agent responsibilities

| Component | Responsibility |
| --- | --- |
| **Coordinator Agent** | Routes recommendation and comparison requests. |
| **Preference Agent** | Extracts structured preferences and handles follow-up changes. |
| **Search Agent** | Calls the product search and ranking tool. |
| **Recommendation Agent** | Explains recommended products and trade-offs using retrieved catalog data. |
| **Comparison Agent** | Retrieves specific product IDs and compares their specifications. |

The recommendation workflow is implemented as a Google ADK `SequentialAgent`. The coordinator uses an ADK `LlmAgent` to delegate between workflows.

## Technology stack

| Technology | Use |
| --- | --- |
| **Google ADK 2.6.2** | Agent orchestration and tool integration |
| **Gemini 2.5 Flash** | Preference extraction, routing, and response generation |
| **Gemini Embedding (`gemini-embedding-001`)** | Dense semantic embeddings |
| **Vertex AI** | Google Cloud model access |
| **Qdrant** | Dense and sparse vector storage and retrieval |
| **BM25 / FastEmbed** | Sparse keyword retrieval |
| **Reciprocal Rank Fusion (RRF)** | Combining dense and sparse search results |
| **Pydantic** | Structured preference validation |
| **Docker** | Local Qdrant deployment |
| **Pytest** | Unit testing |

## Recommendation pipeline

### 1. Extract preferences

For example:

> Find me a gaming laptop under 1500 with at least 16 GB RAM.

The preference agent extracts a maximum price of 1500, a minimum of 16 GB RAM, and gaming as the intended use.

A follow-up such as:

> Actually, make it at least 32 GB RAM. And preferably Lenovo.

updates the RAM requirement, retains the existing budget, and adds Lenovo as a **soft preference**.

### 2. Retrieve candidates

The search tool uses two retrieval methods:

1. **Dense search:** A Gemini embedding represents the query semantically.
2. **Sparse search:** BM25 emphasizes exact words and technical specifications.

Qdrant combines the retrieval results with **Reciprocal Rank Fusion**.

### 3. Enforce constraints and rank

The ranking stage separates:

- **Hard constraints**, such as a maximum price, minimum RAM, required brand, or required touchscreen. Products violating these requirements are ineligible.
- **Soft preferences**, such as a preferred brand, CPU, or GPU. These influence the ordering of eligible products but do not override hard constraints.

For example, a Lenovo laptop outside the user's budget should not be recommended simply because Lenovo is preferred.

### 4. Explain or compare products

The recommendation agent explains the selected products using their catalog attributes. Comparison requests follow a separate path that retrieves products by catalog ID.

## Dataset

The project uses the public Hugging Face dataset:

**[Santhoshwar2204/laptop](https://huggingface.co/datasets/Santhoshwar2204/laptop)**

The prepared local catalog contains approximately **2,160 laptop records**, including fields such as brand, model, CPU, RAM, storage, GPU, screen size, touchscreen status, and price.

**Data limitations:** This is a historical/static catalog. Prices should not be treated as current offers, and some attributes are missing. In particular, missing GPU information can limit hardware-specific filtering and ranking.

Generated raw data, processed records, and embedding caches are excluded from Git by `.gitignore` and must be recreated locally.

## Repository structure

```text
app/
  __init__.py
  retrieval.py
  ranking.py
  schemas.py
  recommender/
    __init__.py
    agent.py
    workflow.py
    tools.py
    agents/
      __init__.py
      preference.py
      search.py
      recommendation.py

scripts/
  prepare_data.py
  create_collection.py
  index_products.py

tests/
  test_ranking.py

evals/
  recommendation.evalset.json

create_eval_set.py
requirements.txt
README.md
.gitignore
```

## Getting started

### Prerequisites

- **Python 3.11** (the development environment used for this project)
- **Docker**, for running Qdrant locally
- A **Google Cloud project** with Vertex AI access to the Gemini models used by the project
- **Google Cloud CLI**, for local Application Default Credentials (ADC)

Google Cloud model usage may incur charges and is subject to service quotas.

### 1. Clone the repository

```bash
git clone https://github.com/majid-lotfian/multi-agent-recommender.git
cd multi-agent-recommender
```

### 2. Create an environment and install dependencies

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

### 3. Configure Vertex AI

Authenticate with Google Cloud:

```bash
gcloud auth application-default login
```

Create a `.env` file in the project root:

```dotenv
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_PROJECT=your-google-cloud-project-id
GOOGLE_CLOUD_LOCATION=global
```

Replace `your-google-cloud-project-id` with your own project ID. The `.env` file is ignored by Git and should not be committed.

### 4. Start Qdrant

```bash
docker run -d --name recommender-qdrant -p 6333:6333 qdrant/qdrant
```

If you have already created the container, start it instead:

```bash
docker start recommender-qdrant
```

Qdrant should be accessible at `http://localhost:6333`.

### 5. Prepare and index the catalog

Run these scripts in order from the repository root:

```bash
python scripts/prepare_data.py
python scripts/create_collection.py
python scripts/index_products.py
```

These steps prepare the catalog, configure the Qdrant collection, and index product representations. The indexing process calls a Google embedding model and can take time or incur API charges.

### 6. Launch the ADK web interface

**Windows PowerShell:**

```powershell
$env:PYTHONPATH = (Get-Location).Path
adk web app
```

**macOS / Linux:**

```bash
PYTHONPATH="$(pwd)" adk web app
```

Open the local address displayed by ADK Web and select the recommender application.

## Example conversations

**Initial recommendation**

> Find me a gaming laptop under 1500 with at least 16 GB RAM.

**Update a hard constraint**

> Actually, make it at least 32 GB RAM.

**Add a soft preference**

> And preferably Lenovo.

**Compare catalog products**

> Compare laptops with catalog IDs 1308 and 1311.

The example IDs refer to records in the prepared catalog. Results depend on the indexed dataset and available model services.

## Tests and evaluation

Run the deterministic ranking tests:

```bash
python -m pytest tests/test_ranking.py -v
```

The existing ranking test suite has **11 passing tests** in the development environment.

The repository also includes preliminary ADK evaluation scenarios for:

- A basic recommendation request
- Multi-turn preference updates
- Product comparison

**Important evaluation caveat:** These ADK scenarios contain user inputs but do not include sufficient reference responses or tool-call trajectories for meaningful default ADK evaluation scores. A previous evaluation run executed the workflows but reported failing scores because the required reference data was absent. Those scores should **not** be interpreted as measured recommendation quality or proof of incorrect recommendations.

### What has and has not been evaluated

| Area | Status |
| --- | --- |
| Deterministic constraint/ranking logic | Covered by unit tests |
| End-to-end workflow execution | Manually exercised |
| Multi-turn preference updates | Manually exercised |
| Product comparisons | Manually exercised |
| Relevance metrics (Precision@K, NDCG@K) | Not established: no labeled relevance benchmark |
| Production latency, cost, and reliability | Not systematically benchmarked |
| Online engagement/conversion | Not available |

## Limitations

- **Static catalog:** No real-time prices, inventory, or seller data.
- **Incomplete attributes:** Some product fields are missing.
- **Heuristic ranking:** Weights are manually defined rather than learned from clicks, purchases, or relevance judgments.
- **Limited evaluation:** No validated ground-truth benchmark for recommendation relevance.
- **Prototype scope:** The system has not been deployed or load-tested as a production recommendation service.

## Possible future work

- Add labeled relevance judgments and compare dense-only, sparse-only, and hybrid retrieval.
- Train or evaluate a learning-to-rank model.
- Incorporate interaction-based personalization and user feedback.
- Add latency, token usage, and cost monitoring.
- Integrate live inventory and price updates.
- Benchmark the multi-agent workflow against a simpler single-agent or deterministic baseline.

## Author

**Majid Lotfian Delouee**  
[GitHub](https://github.com/majid-lotfian)
