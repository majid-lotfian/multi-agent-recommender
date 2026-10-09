
from google.adk.agents import LlmAgent

from app.recommender.workflow import recommendation_workflow
from app.recommender.tools import compare_laptops



comparison_agent = LlmAgent(
    name="ComparisonAgent",
    model="gemini-2.5-flash",
    description=(
        "Compares specific laptop products by catalog ID."
    ),
    instruction="""
    You compare laptops already identified in the catalog.

    When product IDs are available, call compare_laptops
    to retrieve their specifications.

    Compare:
    - CPU
    - GPU
    - RAM
    - Storage
    - Screen
    - Price

    Explain strengths and trade-offs.

    Do not invent specifications.

    If the user refers to previous recommendations,
    use conversation context to identify their IDs.

    If IDs cannot be determined reliably,
    ask the user which products to compare.
    """,
    tools=[compare_laptops],
)


root_agent = LlmAgent(
    name="LaptopCoordinator",
    model="gemini-2.5-flash",
    description="Coordinates laptop search and comparison.",
    instruction="""
    You are the coordinator of a laptop recommendation
    multi-agent system.

    Delegate tasks to the appropriate specialist.

    1. For new laptop searches, recommendations,
       or changes to search preferences:
       delegate to RecommendationWorkflow.

    2. For comparisons of previously identified
       laptop products:
       delegate to ComparisonAgent.

    3. For simple follow-up questions that you
       can answer accurately from conversation context,
       answer directly.

    Do not invent product specifications.

    When a task needs specialist data, delegate
    rather than answering from assumptions.
    """,
    sub_agents=[
        recommendation_workflow,
        comparison_agent,
    ],
)
