from fastapi import APIRouter, Depends, HTTPException, status
from app.middleware.auth import get_current_user
from app.db.supabase_client import get_supabase_client
from app.models.outfit import OutfitCreate, OutfitUpdate, OutfitResponse, OutfitListResponse
from PIL import Image
import io
import httpx
import cloudinary.uploader

router = APIRouter(prefix="/v1/outfits", tags=["outfits"])


def _build_composite_thumbnail(image_urls: list[str], size: int = 512) -> bytes:
    """Download up to 4 garment thumbnails and arrange in a 2x2 grid."""
    urls = image_urls[:4]
    cell = size // 2
    canvas = Image.new("RGB", (size, size), color=(245, 245, 245))
    positions = [(0, 0), (cell, 0), (0, cell), (cell, cell)]
    for i, url in enumerate(urls):
        try:
            resp = httpx.get(url, timeout=10.0)
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            img = img.resize((cell, cell), Image.LANCZOS)
            canvas.paste(img, positions[i])
        except Exception:
            pass  # leave cell blank if download fails
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@router.post("", response_model=OutfitResponse, status_code=status.HTTP_201_CREATED)
async def create_outfit(
    body: OutfitCreate,
    current_user: dict = Depends(get_current_user),
):
    client = get_supabase_client()
    user_id = current_user["user_id"]
    result = (
        client.table("outfits")
        .insert(
            {
                "user_id": user_id,
                "name": body.name,
                "garment_ids": body.garment_ids,
                "occasion": body.occasion,
                "season": body.season,
            }
        )
        .execute()
    )
    return OutfitResponse(**result.data[0])


@router.get("", response_model=OutfitListResponse)
async def list_outfits(current_user: dict = Depends(get_current_user)):
    client = get_supabase_client()
    result = (
        client.table("outfits")
        .select("*", count="exact")
        .eq("user_id", current_user["user_id"])
        .order("created_at", desc=True)
        .execute()
    )
    items = [OutfitResponse(**row) for row in (result.data or [])]
    return OutfitListResponse(items=items, total=result.count or 0)


@router.put("/{outfit_id}", response_model=OutfitResponse)
async def update_outfit(
    outfit_id: str,
    body: OutfitUpdate,
    current_user: dict = Depends(get_current_user),
):
    client = get_supabase_client()
    existing = client.table("outfits").select("user_id").eq("id", outfit_id).maybe_single().execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Outfit not found")
    if existing.data["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Forbidden")

    updates = body.model_dump(exclude_none=True)
    result = client.table("outfits").update(updates).eq("id", outfit_id).execute()
    return OutfitResponse(**result.data[0])


@router.delete("/{outfit_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_outfit(
    outfit_id: str,
    current_user: dict = Depends(get_current_user),
):
    client = get_supabase_client()
    existing = client.table("outfits").select("user_id").eq("id", outfit_id).maybe_single().execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Outfit not found")
    if existing.data["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    client.table("outfits").delete().eq("id", outfit_id).execute()


@router.post("/{outfit_id}/composite", response_model=OutfitResponse)
async def generate_composite(
    outfit_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Generate a collage thumbnail from the outfit's garment thumbnails."""
    client = get_supabase_client()
    outfit_result = client.table("outfits").select("*").eq("id", outfit_id).maybe_single().execute()
    if not outfit_result.data:
        raise HTTPException(status_code=404, detail="Outfit not found")
    outfit = outfit_result.data
    if outfit["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Fetch thumbnail URLs for each garment in the outfit
    garment_result = (
        client.table("garments")
        .select("thumbnail_url")
        .in_("id", outfit["garment_ids"])
        .execute()
    )
    thumb_urls = [r["thumbnail_url"] for r in (garment_result.data or []) if r.get("thumbnail_url")]

    composite_bytes = _build_composite_thumbnail(thumb_urls)
    upload_result = cloudinary.uploader.upload(
        composite_bytes,
        folder=f"drape/{current_user['user_id']}/outfit_composites",
        public_id=outfit_id,
        resource_type="image",
        overwrite=True,
    )
    thumbnail_url: str = upload_result["secure_url"]

    update_result = (
        client.table("outfits")
        .update({"thumbnail_url": thumbnail_url})
        .eq("id", outfit_id)
        .execute()
    )
    return OutfitResponse(**update_result.data[0])
