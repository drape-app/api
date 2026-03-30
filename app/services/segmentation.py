import modal

app = modal.App("wardrobe-segmentation")

@app.function(
    gpu="T4",
    image=modal.Image.debian_slim().pip_install(
        "groundingdino-py", "sam2", "pillow", "torch", "torchvision"
    ),
    timeout=120,
    scaledown_window=60,
)
def segment_garments(image_bytes: bytes, mask_hints: list[str] | None = None) -> list[dict]:
    import torch
    from groundingdino.util.inference import load_model, predict
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import numpy as np
    from PIL import Image
    import io

    clothing_prompts = mask_hints or [
        "shirt", "t-shirt", "blouse", "sweater", "jacket", "coat",
        "jeans", "pants", "shorts", "skirt", "dress", "shoes",
        "sneakers", "boots", "bag", "hat", "scarf", "belt", "socks",
    ]

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    dino_model = load_model(
        "groundingdino/config/cfg_coco.py",
        "weights/groundingdino_swinb_cogcoor.pth",
    )
    text_prompt = " . ".join(clothing_prompts) + " ."
    boxes, logits, phrases = predict(
        model=dino_model,
        image=image_np,
        caption=text_prompt,
        box_threshold=0.35,
        text_threshold=0.25,
    )

    sam2_model = build_sam2(
        "sam2_hiera_large.yaml", "weights/sam2.1_hiera_large.pt"
    )
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image_np)

    results = []
    for box, phrase, score in zip(boxes, phrases, logits):
        masks, scores, _ = predictor.predict(
            box=box.numpy(), multimask_output=False
        )
        if scores[0] > 0.88:
            mask_img = Image.fromarray((masks[0] * 255).astype(np.uint8))
            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")
            results.append(
                {
                    "label": phrase,
                    "mask_png": buf.getvalue(),
                    "bbox": box.tolist(),
                    "confidence": float(score),
                }
            )
    return results
