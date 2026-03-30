import pytest
from app.tasks.process_garment import pick_best_mask, _bbox_area


def _seg(label: str, confidence: float, w: float, h: float) -> dict:
    """Build a minimal segment dict for testing."""
    return {"label": label, "mask_png": b"", "bbox": [0.5, 0.5, w, h], "confidence": confidence}


def test_pick_best_mask_empty_raises():
    with pytest.raises(ValueError, match="No clothing detected"):
        pick_best_mask([])


def test_pick_best_mask_single_item():
    seg = _seg("shirt", 0.9, 0.4, 0.6)
    assert pick_best_mask([seg]) is seg


def test_pick_best_mask_prefers_large_over_tiny_confident():
    # 0.99 * (0.05*0.05) = 0.002475 vs 0.85 * (0.8*0.9) = 0.612 → large wins
    tiny_confident = _seg("button", 0.99, 0.05, 0.05)
    large_moderate = _seg("shirt",  0.85, 0.80, 0.90)
    assert pick_best_mask([tiny_confident, large_moderate]) is large_moderate


def test_pick_best_mask_prefers_confident_when_sizes_equal():
    low  = _seg("shirt",  0.70, 0.5, 0.5)
    high = _seg("jacket", 0.95, 0.5, 0.5)
    assert pick_best_mask([low, high]) is high


def test_bbox_area_normalized():
    assert _bbox_area([0.5, 0.5, 0.4, 0.6]) == pytest.approx(0.24)
