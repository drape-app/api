from supabase import create_client, Client
from app.config import settings
import functools

@functools.lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    return create_client(settings.supabase_url, settings.supabase_service_key)

async def update_garment_status(garment_id: str, status: str, progress_pct: int = 0) -> None:
    """Update sync_status and optional progress metadata on a garment row."""
    client = get_supabase_client()
    payload = {"sync_status": status, "updated_at": "now()"}
    if progress_pct:
        # Store progress in a transient field via notes until dedicated column added
        payload["notes"] = f"__progress:{progress_pct}"
    client.table("garments").update(payload).eq("id", garment_id).execute()

async def update_garment_complete(garment_id: str, data: dict) -> None:
    """Write all AI-extracted fields and set status to SYNCED."""
    client = get_supabase_client()
    payload = {
        **data,
        "sync_status": "SYNCED",
        "updated_at": "now()",
    }
    client.table("garments").update(payload).eq("id", garment_id).execute()

async def get_garment(garment_id: str) -> dict | None:
    """Fetch a single garment row by id."""
    client = get_supabase_client()
    result = client.table("garments").select("*").eq("id", garment_id).maybe_single().execute()
    return result.data if result else None

async def delete_garment(garment_id: str) -> None:
    client = get_supabase_client()
    client.table("garments").delete().eq("id", garment_id).execute()
