import dramatiq
from dramatiq.brokers.redis import RedisBroker
import os
import httpx
import cloudinary
import cloudinary.uploader
import io
import asyncio
from PIL import Image

import modal

# Look up the deployed Modal function by name so this works outside Modal's runtime.
_segment_garments = modal.Function.from_name("wardrobe-segmentation", "segment_garments")
from app.services.metadata import extract_metadata
from app.services.embedding import generate_embedding
from app.db.supabase_client import (
    get_supabase_client,
    update_garment_status,
    update_garment_complete,
)

broker = RedisBroker(url=os.environ["REDIS_URL"])
dramatiq.set_broker(broker)

# Configure Cloudinary from CLOUDINARY_URL env var (parsed automatically)
cloudinary.config(cloudinary_url=os.environ["CLOUDINARY_URL"])


def _bbox_area(bbox: list[float]) -> float:
    """Return normalized area from [cx, cy, w, h] bounding box."""
    return bbox[2] * bbox[3]


def pick_best_mask(segments: list[dict]) -> dict:
    """Select the best clothing mask scored by confidence × bounding-box area.

    Prefers large, confident regions over tiny high-confidence artefacts
    (e.g. a button with score 0.99 loses to a shirt with 0.85 over a large area).

    Raises ValueError when no segments are detected so the caller can mark
    the garment as FAILED rather than silently falling back to the raw image.
    """
    if not segments:
        raise ValueError("No clothing detected in image")
    return max(segments, key=lambda s: s["confidence"] * _bbox_area(s["bbox"]))


def _apply_mask_and_crop(original_bytes: bytes, mask_png: bytes, bbox: list[float]) -> bytes:
    """Composite mask over original, crop to bounding box, return PNG bytes."""
    original = Image.open(io.BytesIO(original_bytes)).convert("RGBA")
    mask = Image.open(io.BytesIO(mask_png)).convert("L").resize(original.size)
    # Zero out pixels outside the mask
    r, g, b, _ = original.split()
    masked = Image.merge("RGBA", (r, g, b, mask))
    # Crop to bounding box (bbox in [cx, cy, w, h] normalised format from DINO)
    W, H = original.size
    cx, cy, bw, bh = bbox
    x1 = int((cx - bw / 2) * W)
    y1 = int((cy - bh / 2) * H)
    x2 = int((cx + bw / 2) * W)
    y2 = int((cy + bh / 2) * H)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    cropped = masked.crop((x1, y1, x2, y2))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()


def _upload_to_cloudinary(image_bytes: bytes, public_id: str, folder: str) -> str:
    """Upload bytes to Cloudinary with auto format/quality and return secure URL."""
    result = cloudinary.uploader.upload(
        image_bytes,
        public_id=public_id,
        folder=folder,
        transformation=[{"fetch_format": "auto", "quality": "auto"}],
        overwrite=True,
        resource_type="image",
    )
    return result["secure_url"]


def _generate_thumbnail(image_bytes: bytes, size: tuple[int, int] = (256, 256)) -> bytes:
    """Resize image to thumbnail dimensions, return JPEG bytes."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail(size, Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@dramatiq.actor(
    queue_name="garment_processing",
    max_retries=5,
    min_backoff=10_000,
    max_backoff=300_000,
)
def process_garment(
    garment_id: str,
    user_id: str,
    image_url: str,
    mask_hints: list[str] | None = None,
) -> None:
    """
    Full garment processing pipeline:
      1. Download original image
      2. Cloud segmentation via Modal (T4 GPU)
      3. For each segment: apply mask, crop, upload segmented PNG to Cloudinary
      4. Extract garment metadata via Gemini 2.5 Flash
      5. Generate 512-dim embedding via Marqo-FashionSigLIP
      6. Upload thumbnail to Cloudinary
      7. Update Supabase row → triggers Realtime push to Android client
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # --- Step 1: Mark as PROCESSING ---
        loop.run_until_complete(update_garment_status(garment_id, "PROCESSING", progress_pct=5))

        # --- Step 2: Download original image ---
        response = httpx.get(image_url, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
        original_bytes: bytes = response.content
        loop.run_until_complete(update_garment_status(garment_id, "PROCESSING", progress_pct=15))

        # --- Step 3: Cloud segmentation via Modal ---
        segments: list[dict] = _segment_garments.remote(original_bytes, mask_hints)
        loop.run_until_complete(update_garment_status(garment_id, "PROCESSING", progress_pct=40))

        # --- Step 4: Pick best segment ---
        best_seg = pick_best_mask(segments)   # raises ValueError → caught below → FAILED
        best_segment_bytes = _apply_mask_and_crop(
            original_bytes, best_seg["mask_png"], best_seg["bbox"]
        )
        folder = f"drape/{user_id}/segmented"
        segmented_image_url = _upload_to_cloudinary(
            best_segment_bytes,
            public_id=f"{garment_id}_segmented",
            folder=folder,
        )
        loop.run_until_complete(update_garment_status(garment_id, "PROCESSING", progress_pct=55))

        # --- Step 5: Extract metadata via Gemini ---
        metadata: dict = loop.run_until_complete(extract_metadata(best_segment_bytes))
        loop.run_until_complete(update_garment_status(garment_id, "PROCESSING", progress_pct=70))

        # --- Step 6: Generate embedding via Marqo-FashionSigLIP ---
        embedding: list[float] = generate_embedding(best_segment_bytes)
        loop.run_until_complete(update_garment_status(garment_id, "PROCESSING", progress_pct=85))

        # --- Step 7: Generate + upload thumbnail ---
        thumb_bytes = _generate_thumbnail(best_segment_bytes)
        thumbnail_url = _upload_to_cloudinary(
            thumb_bytes,
            public_id=f"{garment_id}_thumb",
            folder=f"drape/{user_id}/thumbnails",
        )
        loop.run_until_complete(update_garment_status(garment_id, "PROCESSING", progress_pct=95))

        # --- Step 8: Write all fields to Supabase → Realtime fires ---
        complete_data = {
            "category": metadata.get("category", "unknown"),
            "subcategory": metadata.get("subcategory"),
            "colors": metadata.get("colors", []),
            "pattern": metadata.get("pattern"),
            "fabric": metadata.get("fabric"),
            "brand": metadata.get("brand"),
            "season": metadata.get("season", []),
            "occasions": metadata.get("occasions", []),
            "care_instructions": metadata.get("care_instructions"),
            "style_tags": metadata.get("style_tags", []),
            "formality_score": metadata.get("formality_score"),
            "segmented_image_url": segmented_image_url,
            "thumbnail_url": thumbnail_url,
            "embedding": embedding,
            "notes": None,  # clear the __progress marker
        }
        loop.run_until_complete(update_garment_complete(garment_id, complete_data))

    except Exception as exc:
        loop.run_until_complete(update_garment_status(garment_id, "FAILED"))
        raise exc  # re-raise so Dramatiq can apply retry backoff
    finally:
        loop.close()
