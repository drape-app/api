from fastapi import APIRouter, Depends, Query, HTTPException
from app.middleware.auth import get_current_user
from app.db.supabase_client import get_supabase_client
from app.models.garment import (
    WardrobeListResponse,
    GarmentResponse,
    SimilarRequest,
    SimilarResponse,
    SimilarItem,
    WardrobeStatsResponse,
    OutfitSuggestionRequest,
    OutfitSuggestionResponse,
)
from typing import Any

router = APIRouter(prefix="/v1/wardrobe", tags=["wardrobe"])


@router.get("", response_model=WardrobeListResponse)
async def list_wardrobe(
    category: str | None = Query(default=None),
    season: str | None = Query(default=None),
    occasion: str | None = Query(default=None),
    search: str | None = Query(default=None),
    sort: str = Query(default="created_at_desc"),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    current_user: dict = Depends(get_current_user),
):
    """Return paginated, filtered garments for the authenticated user."""
    client = get_supabase_client()
    user_id = current_user["user_id"]

    query = client.table("garments").select("*", count="exact").eq("user_id", user_id)

    if category:
        query = query.eq("category", category)
    if season:
        query = query.contains("season", [season])
    if occasion:
        query = query.contains("occasions", [occasion])
    if search:
        # Basic ilike search on category + subcategory + style_tags text
        query = query.ilike("category", f"%{search}%")

    # Sorting
    sort_map = {
        "created_at_desc": ("created_at", False),
        "created_at_asc": ("created_at", True),
        "worn_count_desc": ("worn_count", False),
        "formality_asc": ("formality_score", True),
        "formality_desc": ("formality_score", False),
    }
    col, asc = sort_map.get(sort, ("created_at", False))
    query = query.order(col, desc=not asc)

    offset = (page - 1) * limit
    query = query.range(offset, offset + limit - 1)

    result = query.execute()
    total = result.count or 0
    items = [GarmentResponse(**row) for row in (result.data or [])]

    return WardrobeListResponse(
        items=items,
        total=total,
        page=page,
        limit=limit,
        has_more=(offset + len(items)) < total,
    )


@router.post("/similar", response_model=SimilarResponse)
async def find_similar(
    body: SimilarRequest,
    current_user: dict = Depends(get_current_user),
):
    """Find visually similar garments using pgvector cosine similarity."""
    client = get_supabase_client()
    # Call the SQL function defined in the migration
    result = client.rpc(
        "similar_garments",
        {"target_id": body.item_id, "result_limit": body.limit},
    ).execute()
    items = [SimilarItem(id=row["id"], similarity=row["similarity"]) for row in (result.data or [])]
    return SimilarResponse(items=items)


@router.get("/stats", response_model=WardrobeStatsResponse)
async def wardrobe_stats(current_user: dict = Depends(get_current_user)):
    """Return aggregate wardrobe statistics for the user."""
    client = get_supabase_client()
    user_id = current_user["user_id"]

    all_result = client.table("garments").select("category, worn_count, is_favorite").eq("user_id", user_id).execute()
    rows: list[dict[str, Any]] = all_result.data or []

    total = len(rows)
    categories: dict[str, int] = {}
    favorite_count = 0
    for r in rows:
        categories[r["category"]] = categories.get(r["category"], 0) + 1
        if r.get("is_favorite"):
            favorite_count += 1

    sorted_worn = sorted(rows, key=lambda r: r.get("worn_count", 0), reverse=True)
    most_worn = [{"id": r.get("id"), "worn_count": r.get("worn_count", 0)} for r in sorted_worn[:5]]
    least_worn = [{"id": r.get("id"), "worn_count": r.get("worn_count", 0)} for r in sorted_worn[-5:]]

    return WardrobeStatsResponse(
        total_garments=total,
        categories=categories,
        most_worn=most_worn,
        least_worn=least_worn,
        favorite_count=favorite_count,
    )


@router.post("/outfit-suggestion", response_model=OutfitSuggestionResponse)
async def suggest_outfit(
    body: OutfitSuggestionRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Suggest a complete outfit for a given occasion/season.
    Anchors on partial_items if provided; selects best-matching complements
    from the user's wardrobe using formality score + occasion/season filters.
    """
    client = get_supabase_client()
    user_id = current_user["user_id"]

    # Fetch wardrobe filtered by occasion + season
    query = (
        client.table("garments")
        .select("id, category, formality_score, occasions, season, style_tags")
        .eq("user_id", user_id)
        .eq("sync_status", "SYNCED")
        .contains("occasions", [body.occasion])
    )
    if body.season:
        query = query.contains("season", [body.season])

    result = query.execute()
    candidates: list[dict] = result.data or []

    partial_ids = set(body.partial_items or [])
    anchors = [c for c in candidates if c["id"] in partial_ids]
    rest = [c for c in candidates if c["id"] not in partial_ids]

    # Determine target formality from anchors, else default to 5
    if anchors:
        avg_formality = sum(a.get("formality_score") or 5 for a in anchors) / len(anchors)
    else:
        avg_formality = 5.0

    # Greedily fill one item per missing category
    needed_categories = {"tops", "bottoms", "footwear", "outerwear"}
    covered = {a["category"] for a in anchors}
    selected = list(anchors)

    for cat in needed_categories - covered:
        cat_candidates = [c for c in rest if c["category"] == cat]
        if cat_candidates:
            # Pick closest formality score
            best = min(cat_candidates, key=lambda c: abs((c.get("formality_score") or 5) - avg_formality))
            selected.append(best)

    garment_ids = [s["id"] for s in selected]
    explanation = (
        f"Outfit assembled for {body.occasion}"
        + (f" in {body.season}" if body.season else "")
        + f" with formality ~{avg_formality:.0f}/10."
    )

    return OutfitSuggestionResponse(garment_ids=garment_ids, explanation=explanation)
