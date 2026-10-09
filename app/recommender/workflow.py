
from google.adk.agents import SequentialAgent

from app.recommender.agents.preference import preference_agent
from app.recommender.agents.search import search_agent
from app.recommender.agents.recommendation import recommendation_agent


recommendation_workflow = SequentialAgent(
    name="RecommendationWorkflow",
    description=(
        "Finds laptop recommendations by extracting "
        "preferences, searching the catalog, and "
        "explaining ranked results."
    ),
    sub_agents=[
        preference_agent,
        search_agent,
        recommendation_agent,
    ],
)
