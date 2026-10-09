
from pydantic import BaseModel, Field


class LaptopPreferences(BaseModel):
    intended_use: str = "general"

    min_price: float | None = None
    max_price: float | None = None

    min_ram_gb: int | None = None
    max_ram_gb: int | None = None

    min_storage_gb: int | None = None
    max_storage_gb: int | None = None

    min_screen_inches: float | None = None
    max_screen_inches: float | None = None

    required_brand: str | None = None
    preferred_brand: str | None = None

    required_cpu: str | None = None
    preferred_cpu: str | None = None

    required_gpu: str | None = None
    preferred_gpu: str | None = None

    touchscreen_required: bool | None = None

    other_preferences: list[str] = Field(default_factory=list)
