
from google.adk.agents import LlmAgent

from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5,
    initial_delay=2,
    max_delay=30,
    exp_base=2,
    http_status_codes=[429, 500, 502, 503, 504],
)





recommendation_agent = LlmAgent(
    name="RecommendationAgent",
    model="gemini-2.5-flash",
    description="Explains ranked product recommendations.",
    instruction="""
    You are a laptop recommendation specialist.

    User preferences:
    {preferences}

    Search and ranking results:
    {search_results}

    Recommend up to 3 laptops.

    For each laptop:
    - Product ID and name
    - Catalog price
    - CPU, RAM, GPU and storage
    - Why it matches the user
    - Important trade-offs

    Explain the best overall choice.

    Use only products returned by the search tool.

    Never invent specifications, performance
    benchmarks, prices or availability.

    Prices are historical dataset values,
    not verified live retailer prices.

    If no suitable products were found,
    explain that clearly.
    """,
    generate_content_config=types.GenerateContentConfig(
        http_options=types.HttpOptions(
            retry_options=retry_config,
        ),
    ),
)
