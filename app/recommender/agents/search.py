
from google.adk.agents import LlmAgent

from app.recommender.tools import search_and_rank_laptops

from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5,
    initial_delay=2,
    max_delay=30,
    exp_base=2,
    http_status_codes=[429, 500, 502, 503, 504],
)




search_agent = LlmAgent(
    name="SearchAgent",
    model="gemini-2.5-flash",
    description="Searches and ranks real laptops.",
    instruction="""
    You are the product search specialist.

    The extracted preferences are:
    {preferences}

    Call search_and_rank_laptops exactly once.

    Pass the preferences as a JSON string using
    the preferences_json argument.

    Use limit=5.

    Return the actual tool results.
    Never invent products or prices.
    """,
    tools=[search_and_rank_laptops],
    output_key="search_results",
    generate_content_config=types.GenerateContentConfig(
        http_options=types.HttpOptions(
            retry_options=retry_config,
        ),
    ),
)
