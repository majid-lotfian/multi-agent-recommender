
from google.adk.agents import LlmAgent

from app.schemas import LaptopPreferences
from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5,
    initial_delay=2,
    max_delay=30,
    exp_base=2,
    http_status_codes=[429, 500, 502, 503, 504],
)




preference_agent = LlmAgent(
    name="PreferenceAgent",
    model="gemini-2.5-flash",
    description=(
        "Extracts and updates laptop requirements "
        "across conversation turns."
    ),
    instruction="""
    You manage the user's laptop preferences.

    Previously saved preferences:
    {preferences?}

    Read the user's latest request and update
    the saved preferences.

    RULES:

    1. Preserve existing preferences unless the user
       explicitly changes or removes them.

    2. When the user changes a requirement,
       replace only the relevant field.

       Example:
       Previous min_ram_gb = 16
       User: "Actually, make it 32 GB"
       Result: min_ram_gb = 32

    3. When the user adds a preference,
       preserve all unrelated fields.

       Example:
       User: "I prefer Lenovo"
       Set preferred_brand = "Lenovo"

    4. When the user removes a constraint,
       set the corresponding field to null.

       Example:
       User: "Remove my budget limit"
       Set min_price = null
       Set max_price = null

    5. Distinguish hard requirements from preferences.

       "Must be Lenovo" -> required_brand
       "Prefer Lenovo" -> preferred_brand

    6. Never invent missing requirements.

    7. If the user starts an entirely new search
       with clearly unrelated requirements,
       reset the previous preferences.

    8. Return the COMPLETE updated preference object,
       not just the changed fields.
    """,
    output_schema=LaptopPreferences,
    output_key="preferences",
    generate_content_config=types.GenerateContentConfig(
      http_options=types.HttpOptions(
         retry_options=retry_config,
      ),
   ),
)
