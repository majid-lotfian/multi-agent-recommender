
import json
from pathlib import Path

from google.adk.evaluation.eval_case import (
    EvalCase,
    Invocation,
    SessionInput,
)
from google.adk.evaluation.eval_set import EvalSet
from google.genai import types


def user_turn(message: str) -> Invocation:
    return Invocation(
        user_content=types.Content(
            role="user",
            parts=[types.Part(text=message)],
        )
    )


cases = [
    EvalCase(
        eval_id="basic_recommendation",
        conversation=[
            user_turn(
                "Find me a gaming laptop under 1500 "
                "with at least 16 GB RAM."
            )
        ],
    ),
    EvalCase(
        eval_id="preference_updates",
        conversation=[
            user_turn(
                "Find me a gaming laptop under 1500 "
                "with at least 16 GB RAM."
            ),
            user_turn(
                "Actually, make it at least 32 GB RAM."
            ),
            user_turn(
                "And preferably Lenovo."
            ),
        ],
    ),
    EvalCase(
        eval_id="product_comparison",
        conversation=[
            user_turn(
                "Compare laptops with catalog IDs "
                "1308 and 1311."
            )
        ],
    ),
]

eval_set = EvalSet(
    eval_set_id="laptop_recommendation_v1",
    name="Laptop recommendation evaluation",
    description=(
        "End-to-end recommendation, preference "
        "memory, and product comparison scenarios."
    ),
    eval_cases=cases,
)

Path("evals").mkdir(exist_ok=True)

output = Path("evals/recommendation.evalset.json")

output.write_text(
    json.dumps(
        eval_set.model_dump(
            mode="json",
            exclude_none=True,
        ),
        indent=2,
    ),
    encoding="utf-8",
)

print(f"Created: {output}")
print(f"Evaluation cases: {len(cases)}")
