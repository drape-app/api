import json
import os
from PIL import Image
import io

# Lazy init — configure only when first used so module can be imported in tests
_genai = None
_model = None


def _get_model():
    global _genai, _model
    if _model is None:
        import google.generativeai as genai  # lazy import — heavy dep
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        _genai = genai
        _model = genai.GenerativeModel("gemini-2.5-flash")
    return _model, _genai

METADATA_PROMPT = """You are an expert fashion analyst. Analyze this garment image and return a single JSON object with the following fields. Be precise and concise.

{
  "category": "string — primary garment category, one of: tops, bottoms, dresses, outerwear, footwear, accessories, activewear, swimwear, underwear, formalwear",
  "subcategory": "string — specific type, e.g. 't-shirt', 'jeans', 'blazer', 'sneakers', 'tote bag'",
  "colors": [
    {
      "name": "string — common color name, e.g. 'navy blue', 'off-white', 'forest green'",
      "hex": "string — best-guess hex code, e.g. '#1B2A4A'",
      "proportion": "number — fraction of garment this color covers, 0.0–1.0"
    }
  ],
  "pattern": "string or null — one of: solid, stripes, plaid, floral, geometric, animal_print, abstract, logo, colorblock, polka_dots, camouflage, paisley, houndstooth, null",
  "fabric": "string or null — primary fabric, e.g. 'cotton', 'denim', 'wool', 'polyester', 'linen', 'silk', 'leather', 'synthetic blend'",
  "brand": "string or null — visible brand name/logo if detectable, otherwise null",
  "season": ["string"] ,
  "occasions": ["string"],
  "care_instructions": "string or null — inferred care advice, e.g. 'machine wash cold, tumble dry low'",
  "style_tags": ["string"],
  "formality_score": "integer 0–10 — 0=athletic/loungewear, 5=smart casual, 10=black tie"
}

For 'season': choose any combination from ['spring', 'summer', 'fall', 'winter'].
For 'occasions': choose any combination from ['casual', 'work', 'formal', 'athletic', 'outdoor', 'beach', 'party', 'date_night', 'travel'].
For 'style_tags': up to 5 descriptive tags, e.g. ['minimalist', 'oversized', 'vintage', 'streetwear', 'preppy', 'boho', 'classic', 'edgy'].

Return only the raw JSON object — no markdown, no explanation, no code fences."""


async def extract_metadata(image_bytes: bytes) -> dict:
    """Call Gemini 2.5 Flash Vision to extract structured garment metadata."""
    model, genai = _get_model()
    image = Image.open(io.BytesIO(image_bytes))
    response = await model.generate_content_async(
        [METADATA_PROMPT, image],
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    return json.loads(response.text)
