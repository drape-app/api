import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.middleware.auth import get_current_user


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_USER = {"user_id": "test-user-uuid-1234", "email": "test@drape.app"}
FAKE_GARMENT_ID = "garment-uuid-5678"
FAKE_ORIGINAL_URL = "https://res.cloudinary.com/test/image/upload/original.jpg"


def override_get_current_user():
    return FAKE_USER


app.dependency_overrides[get_current_user] = override_get_current_user


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Helper mocks
# ---------------------------------------------------------------------------

def _mock_cloudinary_upload(image_bytes, **kwargs):
    return {"secure_url": FAKE_ORIGINAL_URL}


def _mock_supabase_insert():
    mock = MagicMock()
    mock.table.return_value.insert.return_value.execute.return_value.data = [
        {"id": FAKE_GARMENT_ID}
    ]
    return mock


def _mock_supabase_garment_row(sync_status="PROCESSING", notes="__progress:40"):
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = {
        "id": FAKE_GARMENT_ID,
        "user_id": FAKE_USER["user_id"],
        "sync_status": sync_status,
        "notes": notes,
        "category": "tops",
        "thumbnail_url": None,
    }
    return mock


# ---------------------------------------------------------------------------
# Tests: POST /v1/garments/upload
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_garment_enqueues_task(client):
    """Upload endpoint creates a PENDING row and enqueues the Dramatiq task."""
    fake_image_bytes = b"\x89PNG\r\n" + b"0" * 100  # minimal fake PNG header

    with (
        patch("app.api.garments.cloudinary.uploader.upload", side_effect=_mock_cloudinary_upload),
        patch("app.api.garments.get_supabase_client", return_value=_mock_supabase_insert()),
        patch("app.api.garments.process_garment") as mock_task,
    ):
        mock_task.send = MagicMock()
        response = await client.post(
            "/v1/garments/upload",
            files={"image": ("test.jpg", fake_image_bytes, "image/jpeg")},
        )

    assert response.status_code == 202
    data = response.json()
    assert data["garment_id"] == FAKE_GARMENT_ID
    assert data["status"] == "PENDING"
    mock_task.send.assert_called_once_with(
        FAKE_GARMENT_ID, FAKE_USER["user_id"], FAKE_ORIGINAL_URL, None
    )


@pytest.mark.asyncio
async def test_upload_empty_file_returns_400(client):
    """Upload with empty bytes returns HTTP 400."""
    with (
        patch("app.api.garments.cloudinary.uploader.upload", side_effect=_mock_cloudinary_upload),
        patch("app.api.garments.get_supabase_client", return_value=_mock_supabase_insert()),
        patch("app.api.garments.process_garment"),
    ):
        response = await client.post(
            "/v1/garments/upload",
            files={"image": ("empty.jpg", b"", "image/jpeg")},
        )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Tests: GET /v1/garments/{id}/status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_garment_status_processing(client):
    """Status endpoint returns PROCESSING with progress_pct from notes field."""
    with patch("app.api.garments.get_garment", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "id": FAKE_GARMENT_ID,
            "user_id": FAKE_USER["user_id"],
            "sync_status": "PROCESSING",
            "notes": "__progress:40",
            "category": "tops",
            "thumbnail_url": None,
        }
        response = await client.get(f"/v1/garments/{FAKE_GARMENT_ID}/status")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "PROCESSING"
    assert data["progress_pct"] == 40
    assert data["partial_results"] is None


@pytest.mark.asyncio
async def test_get_garment_status_synced(client):
    """Status endpoint returns SYNCED with partial_results when processing complete."""
    with patch("app.api.garments.get_garment", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "id": FAKE_GARMENT_ID,
            "user_id": FAKE_USER["user_id"],
            "sync_status": "SYNCED",
            "notes": None,
            "category": "tops",
            "thumbnail_url": "https://res.cloudinary.com/test/thumb.jpg",
        }
        response = await client.get(f"/v1/garments/{FAKE_GARMENT_ID}/status")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SYNCED"
    assert data["progress_pct"] == 0
    assert data["partial_results"]["category"] == "tops"


@pytest.mark.asyncio
async def test_get_garment_status_not_found(client):
    """Status endpoint returns 404 for unknown garment ID."""
    with patch("app.api.garments.get_garment", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = None
        response = await client.get(f"/v1/garments/nonexistent-id/status")

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Tests: POST /v1/wardrobe/similar
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_similar_garments(client):
    """Similar endpoint calls the pgvector SQL function and returns ranked items."""
    mock_supabase = MagicMock()
    mock_supabase.rpc.return_value.execute.return_value.data = [
        {"id": "uuid-a", "similarity": 0.97},
        {"id": "uuid-b", "similarity": 0.91},
    ]
    with patch("app.api.wardrobe.get_supabase_client", return_value=mock_supabase):
        response = await client.post(
            "/v1/wardrobe/similar",
            json={"item_id": FAKE_GARMENT_ID, "limit": 2},
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 2
    assert data["items"][0]["similarity"] == pytest.approx(0.97, rel=1e-3)
    mock_supabase.rpc.assert_called_once_with(
        "similar_garments",
        {"target_id": FAKE_GARMENT_ID, "result_limit": 2},
    )
