"""Text extraction from a scanned document image using Tesseract OCR."""
import pytesseract


def extract_text(image, lang="eng", psm=6):
    """Run OCR on `image` and return the extracted text.

    psm=6 (assume a single uniform block of text) is used by default
    instead of Tesseract's automatic page-segmentation mode, which tends
    to misfire on a single cropped scan (e.g. the thresholded border can
    get mistaken for a layout element, returning no text at all).
    """
    try:
        return pytesseract.image_to_string(image, lang=lang, config=f"--psm {psm}")
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract is not installed or not in PATH. "
            "See the README for installation instructions."
        ) from exc


def ocr_confidence(image, lang="eng", psm=6):
    """Return Tesseract's mean word-level confidence (0-100) for `image`,
    or None if no text was recognized at all.

    This doesn't tell you the text is *correct* - only how confident
    Tesseract is in what it read. A low score reliably flags garbage
    (wrong rotation, no real document found, blank/noise image) and is
    cheap enough to run on every document in a batch automatically,
    instead of someone having to open every .txt file by hand to check.
    """
    data = pytesseract.image_to_data(image, lang=lang, config=f"--psm {psm}",
                                      output_type=pytesseract.Output.DICT)
    scores = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit() and int(c) >= 0]
    if not scores:
        return None
    return sum(scores) / len(scores)
