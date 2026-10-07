"""Correct 90/180/270-degree rotation on an already-isolated document crop.

Orientation detection (OSD) is run per-document, on a single perspective-
corrected crop, rather than on the full multi-document photo - that's the
difference that makes it reliable: Tesseract's OSD gets confused when a
frame holds several documents at different angles at once, but is quite
reliable on one clean, isolated document image.
"""
import cv2
import pytesseract

# Maps Tesseract's reported "rotate" value (degrees needed to make the
# text upright) to the matching OpenCV rotation flag.
_ROTATION_FLAGS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _try_osd(image):
    try:
        return pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
    except (pytesseract.TesseractError, RuntimeError):
        return None


def auto_deskew(image, min_confidence=1.0):
    """Rotate `image` by a multiple of 90 degrees to make its text
    upright, based on Tesseract's orientation detection.

    Returns (possibly-rotated image, rotation applied in degrees). If OSD
    can't confidently determine an orientation, the image is returned
    unchanged with rotation 0 - this is a deliberately conservative
    fallback, since guessing wrong is worse than doing nothing.
    """
    osd = _try_osd(image)
    if osd is None:
        # OSD can fail outright ("too few characters") on crops where the
        # text is sparse or confined to a small strip of the image - often
        # just a resolution problem. One retry on an upscaled copy recovers
        # a meaningful number of these without much extra cost.
        upscaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        osd = _try_osd(upscaled)
    if osd is None:
        return image, 0

    confidence = osd.get("orientation_conf", 0)
    rotate_by = osd.get("rotate", 0)

    if confidence < min_confidence or rotate_by not in _ROTATION_FLAGS:
        return image, 0

    return cv2.rotate(image, _ROTATION_FLAGS[rotate_by]), rotate_by
