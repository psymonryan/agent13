"""resize_image_uri must survive corrupt/truncated images.

Bug: PIL's Image.open is lazy — it only reads the header. A truncated
PNG (e.g. scp'd while the remote side was still writing it) opens fine,
then raises ``OSError: image file is truncated`` at ``img.resize()``,
which was outside the try/except. The exception propagated up through
``_build_tool_result_content`` into ``_llm_turn`` and killed the whole
agent turn (seen at the end of a 19-hour session, 2026-09-20).

Fix: the entire open/resize/save is wrapped, and any failure returns the
original data URI unchanged so a bad image degrades to a provider-side
API error instead of aborting the turn.
"""

import base64
import io

import pytest
from PIL import Image

from agent13.vision import image_uri_decodable, resize_image_uri


def _uri(img_bytes: bytes, media_type: str = "image/png") -> str:
    return f"data:{media_type};base64," + base64.b64encode(img_bytes).decode()


def _png_bytes(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _decode(uri: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))


def test_truncated_png_returns_unchanged():
    """Header intact, pixel data cut off — the 2026-09-20 failure case."""
    full = _png_bytes(2000, 1500)
    truncated = full[: len(full) // 2]
    uri = _uri(truncated)

    # Sanity: this input really does fail at resize under a plain decode
    img = Image.open(io.BytesIO(truncated))
    with pytest.raises(OSError):
        img.resize((896, 672), Image.LANCZOS)

    assert resize_image_uri(uri, 1568) == uri


def test_truncated_png_small_enough_returns_unchanged():
    """Small truncated image passes the size check without decoding."""
    full = _png_bytes(100, 100)
    uri = _uri(full[: len(full) // 2])
    assert resize_image_uri(uri, 1568) == uri


def test_garbage_data_returns_unchanged():
    uri = "data:image/png;base64,not-an-image"
    assert resize_image_uri(uri, 1568) == uri


def test_large_png_is_resized():
    uri = _uri(_png_bytes(2000, 1500))
    out = resize_image_uri(uri, 1568)
    assert out != uri
    w, h = _decode(out).size
    assert max(w, h) <= 1568
    # Aspect ratio preserved (2000:1500 = 4:3)
    assert abs(w / h - 4 / 3) < 0.01


def test_small_png_is_unchanged():
    uri = _uri(_png_bytes(100, 50))
    assert resize_image_uri(uri, 1568) == uri


def test_large_jpeg_is_resized_keeps_format():
    buf = io.BytesIO()
    Image.new("RGB", (3000, 2000)).save(buf, format="JPEG")
    uri = _uri(buf.getvalue(), media_type="image/jpeg")
    out = resize_image_uri(uri, 1568)
    assert out.startswith("data:image/jpeg;")
    assert max(_decode(out).size) <= 1568


def test_max_dimension_zero_disables_resize():
    uri = _uri(_png_bytes(2000, 1500))
    assert resize_image_uri(uri, 0) == uri


# ── image_uri_decodable ─────────────────────────────────────────────────────


def test_decodable_valid_png():
    assert image_uri_decodable(_uri(_png_bytes(200, 100))) is True


def test_decodable_truncated_png():
    full = _png_bytes(2000, 1500)
    assert image_uri_decodable(_uri(full[: len(full) // 2])) is False


def test_decodable_garbage():
    assert image_uri_decodable("data:image/png;base64,zzz") is False


def test_decodable_not_a_data_uri():
    assert image_uri_decodable("https://example.com/img.png") is False
