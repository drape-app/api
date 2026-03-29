from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from app.middleware.auth import get_current_user
from app.db.supabase_client import get_supabase_client, delete_garment, get_garment
from app.tasks.process_garment import process_garment
from app.models.garment import (
    GarmentUploadResponse,
    GarmentStatusResponse,
    SimilarRequest,
    SimilarResponse,
    SimilarItem,
)
import cloudinary.uploader
import json

router = APIRouter(prefix="/v1/garments", tags=["garments"])


@router.post("/upload", response_model=GarmentUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_garment(
    image: UploadFile = File(...),
    mask_hints: str | None = Form(default=None),
    current_user: dict = Depends(get_current_user),
):
    """Accept a multipart image upload, create a PENDING garment row, enqueue processing."""
    user_id = current_user["user_id"]
    hints: list[str] | None = json.loads(mask_hints) if mask_hints else None

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    # Upload original to Cloudinary to get a stable URL
    upload_result = cloudinary.uploader.upload(
        image_bytes,
        folder=f"drape/{user_id}/originals",
        resource_type="image",
    )
    original_url: str = upload_result["secure_url"]

    # Insert PENDING garment row into Supabase
    client = get_supabase_client()
    insert_result = (
        client.table("garments")
        .insert(
            {
                "user_id": user_id,
                "category": "unknown",  # will be updated by AI pipeline
                "original_image_url": original_url,
                "sync_status": "PENDING",
            }
        )
        .execute()
    )
    garment_id: str = insert_result.data[0]["id"]

    # Enqueue background task
    process_garment.send(garment_id, user_id, original_url, hints)

    return GarmentUploadResponse(garment_id=garment_id)


@router.get("/{garment_id}/status", response_model=GarmentStatusResponse)
async def get_garment_status(
    garment_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Poll processing status and optional partial results."""
    row = await get_garment(garment_id)
    if not row:
        raise HTTPException(status_code=404, detail="Garment not found")
    if row["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Parse progress stored in notes field
    progress = 0
    notes = row.get("notes") or ""
    if notes.startswith("__progress:"):
        try:
            progress = int(notes.split(":")[1])
        except ValueError:
            pass

    partial: dict | None = None
    if row["sync_status"] == "SYNCED":
        partial = {
            "category": row.get("category"),
            "thumbnail_url": row.get("thumbnail_url"),
        }

    return GarmentStatusResponse(
        garment_id=garment_id,
        status=row["sync_status"],
        progress_pct=progress,
        partial_results=partial,
    )


@router.delete("/{garment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_garment(
    garment_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete garment row plus Cloudinary assets."""
    row = await get_garment(garment_id)
    if not row:
        raise HTTPException(status_code=404, detail="Garment not found")
    if row["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Forbidden")

    user_id = current_user["user_id"]
    # Delete Cloudinary assets
    for key_suffix in ["_segmented", "_thumb"]:
        public_id_base = f"drape/{user_id}"
        # Attempt deletion; ignore errors if asset does not exist
        try:
            cloudinary.uploader.destroy(f"{public_id_base}/segmented/{garment_id}{key_suffix}")
            cloudinary.uploader.destroy(f"{public_id_base}/thumbnails/{garment_id}{key_suffix}")
            cloudinary.uploader.destroy(f"{public_id_base}/originals/{garment_id}")
        except Exception:
            pass

    await delete_garment(garment_id)
