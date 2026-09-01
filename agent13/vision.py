"""Vision support: image resizing and sidecar description.

Provides:
- resize_image_uri(): Resize an image data URI to a max dimension
- describe_image(): Send an image to a sidecar vision model, get text back
"""

import base64
import io
import re

# Sidecar prompt template used when no explicit question is provided.
# The main model reads this description and reasons about it.
DEFAULT_SIDECAR_PROMPT = (
    "Context: {tool_text}. "
    "Describe this image in detail: all visible text, UI elements, colors, "
    "layout, and anything notable. Be thorough and specific."
)

# Resize targets per routing mode (longest side in pixels).
# - Sidecar (Gemma 3): native 896x896 processing
# - Native (OpenAI/Anthropic API): server-side resize at 1568px
SIDECAR_MAX_DIMENSION = 896
NATIVE_MAX_DIMENSION = 1568

_DATA_URI_RE = re.compile(r"data:(image/[\w.+-]+);base64,(.+)", re.DOTALL)


def resize_image_uri(data_uri: str, max_dimension: int) -> str:
    """Resize image in a data URI so the longest side <= max_dimension.

    Args:
        data_uri: Data URI string ("data:image/png;base64,...").
        max_dimension: Maximum pixel size for the longest side.
                       0 or negative = no resize (return unchanged).

    Returns:
        New data URI (possibly resized), or the original if no resize needed.
    """
    if max_dimension <= 0:
        return data_uri

    match = _DATA_URI_RE.match(data_uri)
    if not match:
        return data_uri

    media_type, b64_data = match.group(1), match.group(2)

    try:
        from PIL import Image
    except ImportError:
        # Pillow not installed — return unchanged
        return data_uri

    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64_data)))
    except Exception:
        # Not a valid image — return unchanged
        return data_uri

    w, h = img.size
    if max(w, h) <= max_dimension:
        return data_uri  # Already small enough

    # Resize preserving aspect ratio
    scale = max_dimension / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    img = img.resize(new_size, Image.LANCZOS)

    # Re-encode
    buf = io.BytesIO()
    fmt = media_type.split("/")[1].upper()
    if fmt in ("JPEG", "JPG"):
        fmt = "JPEG"
    elif fmt in ("TIF", "TIFF"):
        fmt = "TIFF"
    try:
        img.save(buf, format=fmt)
    except Exception:
        # Fallback to PNG if format not supported for saving
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        media_type = "image/png"

    new_b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:{media_type};base64,{new_b64}"


async def describe_image(
    provider,
    model: str | None,
    image_uri: str,
    prompt: str,
) -> str:
    """Send image to a vision model via sidecar, return text description.

    Args:
        provider: ProviderConfig for the sidecar (e.g. oMLX endpoint).
        model: Model name, or None to use server default.
        image_uri: Data URI ("data:image/png;base64,...").
        prompt: Question or instruction for the vision model.

    Returns:
        Text description from the vision model.
    """
    import httpx

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=provider.api_base,
        api_key=provider.get_api_key() or "unused",
        timeout=httpx.Timeout(provider.connect_timeout, read=provider.read_timeout),
    )

    kwargs = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_uri}},
                ],
            }
        ],
    }
    if model:
        kwargs["model"] = model

    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""
