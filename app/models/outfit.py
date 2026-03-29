from pydantic import BaseModel
from datetime import date, datetime

class OutfitCreate(BaseModel):
    name: str
    garment_ids: list[str]
    occasion: str | None = None
    season: str | None = None

class OutfitUpdate(BaseModel):
    name: str | None = None
    garment_ids: list[str] | None = None
    occasion: str | None = None
    season: str | None = None

class OutfitResponse(BaseModel):
    id: str
    user_id: str
    name: str
    garment_ids: list[str]
    occasion: str | None
    season: str | None
    thumbnail_url: str | None
    worn_count: int
    last_worn_date: date | None
    created_at: datetime

class OutfitListResponse(BaseModel):
    items: list[OutfitResponse]
    total: int
