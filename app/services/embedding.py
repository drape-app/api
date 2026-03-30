from PIL import Image
import io

_processor = None
_model = None


def _load_model():
    global _processor, _model
    if _processor is None:
        from transformers import AutoProcessor, AutoModel  # lazy import — heavy dep
        _processor = AutoProcessor.from_pretrained("Marqo/marqo-fashionSigLIP", trust_remote_code=True)
        _model = AutoModel.from_pretrained("Marqo/marqo-fashionSigLIP", trust_remote_code=True).eval()


def generate_embedding(image_bytes: bytes) -> list[float]:
    """Generate a 512-dim L2-normalised fashion embedding."""
    import torch  # lazy import — heavy dep
    _load_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = _processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        feats = _model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze().tolist()
