from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from app.middleware.auth import get_current_user
from app.db.supabase_client import get_supabase_client, delete_garment, get_garment, update_garment_status
from app.tasks.process_garment import (
    process_garment,
    _apply_mask_and_crop,
    _generate_thumbnail,
    _upload_to_cloudinary,
)
from app.models.garment import (
    GarmentUploadResponse,
    GarmentStatusResponse,
    SimilarRequest,
    SimilarResponse,
    SimilarItem,
    DetectedItem,
    DetectAllResponse,
    ConfirmBatchRequest,
)
from app.services.segmentation import segment_garments
from app.services.metadata import extract_metadata
from app.services.embedding import generate_embedding
import cloudinary.uploader
import asyncio
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


@router.post("/detect-all", response_model=DetectAllResponse)
async def detect_all_garments(
    image: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload a photo, run full multi-item detection (Modal SAM2 + GroundingDINO),
    and persist one DETECTED garment row per detected segment.

    Returns thumbnail URL + metadata for each detected item so the client can
    present a selection UI before confirming.
    """
    user_id = current_user["user_id"]

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    # Upload original to Cloudinary
    upload_result = cloudinary.uploader.upload(
        image_bytes,
        folder=f"drape/{user_id}/originals",
        resource_type="image",
    )
    original_url: str = upload_result["secure_url"]

    # Run cloud segmentation (blocking Modal call — run in thread)
    segments: list[dict] = await asyncio.to_thread(
        segment_garments.remote, image_bytes
    )

    client = get_supabase_client()
    detected_items: list[DetectedItem] = []

    for seg in segments:
        try:
            # Crop + mask this segment
            seg_bytes = _apply_mask_and_crop(image_bytes, seg["mask_png"], seg["bbox"])

            # Extract metadata (async)
            metadata: dict = await extract_metadata(seg_bytes)

            # Generate embedding (sync — run in thread)
            embedding: list[float] = await asyncio.to_thread(generate_embedding, seg_bytes)

            # Thumbnail
            thumb_bytes = _generate_thumbnail(seg_bytes)

            # Insert DETECTED garment row first to get an ID
            insert_result = (
                client.table("garments")
                .insert(
                    {
                        "user_id": user_id,
                        "category": metadata.get("category", "unknown"),
                        "original_image_url": original_url,
                        "sync_status": "DETECTED",
                    }
                )
                .execute()
            )
            garment_id: str = insert_result.data[0]["id"]

            # Upload segmented image + thumbnail to Cloudinary
            segmented_url = _upload_to_cloudinary(
                seg_bytes,
                public_id=f"{garment_id}_segmented",
                folder=f"drape/{user_id}/segmented",
            )
            thumbnail_url = _upload_to_cloudinary(
                thumb_bytes,
                public_id=f"{garment_id}_thumb",
                folder=f"drape/{user_id}/thumbnails",
            )

            # Update row with full metadata
            colors = metadata.get("colors", [])
            client.table("garments").update(
                {
                    "subcategory": metadata.get("subcategory"),
                    "colors": colors,
                    "pattern": metadata.get("pattern"),
                    "fabric": metadata.get("fabric"),
                    "brand": metadata.get("brand"),
                    "season": metadata.get("season", []),
                    "occasions": metadata.get("occasions", []),
                    "care_instructions": metadata.get("care_instructions"),
                    "style_tags": metadata.get("style_tags", []),
                    "formality_score": metadata.get("formality_score"),
                    "segmented_image_url": segmented_url,
                    "thumbnail_url": thumbnail_url,
                    "embedding": embedding,
                    "updated_at": "now()",
                }
            ).eq("id", garment_id).execute()

            detected_items.append(
                DetectedItem(
                    garment_id=garment_id,
                    thumbnail_url=thumbnail_url,
                    category=metadata.get("category", "unknown"),
                    confidence=seg["confidence"],
                    colors=colors,
                )
            )
        except Exception:
            # Skip individual segment failures — don't fail the whole request
            continue

    return DetectAllResponse(items=detected_items)


@router.post("/confirm-batch")
async def confirm_batch(
    request: ConfirmBatchRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Mark selected DETECTED garments as SYNCED and delete the rest.
    Called after the user confirms which items to add to their wardrobe.
    """
    user_id = current_user["user_id"]
    client = get_supabase_client()

    # Mark selected garments as SYNCED
    for gid in request.selected_ids:
        await update_garment_status(gid, "SYNCED")

    # Delete remaining DETECTED garments for this user that were not selected
    excluded = request.selected_ids if request.selected_ids else ["__none__"]
    (
        client.table("garments")
        .delete()
        .eq("user_id", user_id)
        .eq("sync_status", "DETECTED")
        .not_.in_("id", excluded)
        .execute()
    )

    return {"ok": True}
