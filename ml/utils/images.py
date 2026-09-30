"""Image normalisation for sources with large or EXIF-rotated photos."""
from __future__ import annotations

import io

from PIL import Image, ImageOps

JPEG_QUALITY = 92


def normalise_image(data: bytes, expected_size: tuple[int, int], max_side: int) -> bytes:
    """Apply EXIF orientation, check it matches the annotated size, downscale and re-encode as JPEG.

    YOLO labels are normalised to [0, 1], so a uniform resize does not change them;
    only the orientation must match the frame the annotations were made in.
    """
    with Image.open(io.BytesIO(data)) as original:
        image = ImageOps.exif_transpose(original)
        if image.size != expected_size:
            raise ValueError(f"image size {image.size} after EXIF correction != annotated size {expected_size}")
        image = image.convert("RGB")
        scale = max_side / max(image.size)
        if scale < 1:
            new_size = (round(image.width * scale), round(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()
