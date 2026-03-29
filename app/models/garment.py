from pydantic import BaseModel, Field
from typing import Any
import uuid
from datetime import date, datetime

class ColorInfo(BaseModel):
    name: str
    hex: str
    proportion: float = Field(ge=0.0, le=1.0)

class GarmentUploadResponse(BaseModel):
    garment_id: str
    status: str = "PENDING"
    message: str = "Processing started"

class GarmentStatusResponse(BaseModel):
    garment_id: str
    status: str
    progress_pct: int = 0
    partial_results: dict[str, Any] | None = None

class GarmentResponse(BaseModel):
    id: str
    user_id: str
    category: str
    subcategory: str | None
    colors: list[ColorInfo]
    pattern: str | None
    fabric: str | None
    brand: str | None
    season: list[str]
    occasions: list[str]
    care_instructions: str | None
    style_tags: list[str]
    formality_score: int | None
    original_image_url: str | None
    segmented_image_url: str | None
    thumbnail_url: str | None
    sync_status: str
    worn_count: int
    last_worn_date: date | None
    is_favorite: bool
    notes: str | None
    created_at: datetime
    updated_at: datetime

class WardrobeListResponse(BaseModel):
    items: list[GarmentResponse]
    total: int
    page: int
    limit: int
    has_more: bool

class SimilarRequest(BaseModel):
    item_id: str
    limit: int = Field(default=10, ge=1, le=50)

class SimilarItem(BaseModel):
    id: str
    similarity: float

class SimilarResponse(BaseModel):
    items: list[SimilarItem]

class WardrobeStatsResponse(BaseModel):
    total_garments: int
    categories: dict[str, int]
    most_worn: list[dict[str, Any]]
    least_worn: list[dict[str, Any]]
    favorite_count: int

class OutfitSuggestionRequest(BaseModel):
    occasion: str
    season: str | None = None
    partial_items: list[str] | None = None  # garment UUIDs to anchor the outfit

class OutfitSuggestionResponse(BaseModel):
    garment_ids: list[str]
    explanation: str
