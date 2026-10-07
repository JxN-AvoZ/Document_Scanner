"""Clean up a warped document image so it looks like a proper scan."""
import cv2


def enhance_scan(image, mode="adaptive"):
    """Apply a scan-style finish to `image`.

    mode:
      adaptive - classic black & white scanned-paper look (default)
      gray     - grayscale only, no thresholding
      color    - leave the image in color, just return it as-is
    """
    if mode == "color":
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    if mode == "gray":
        return gray

    if mode == "adaptive":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 25, 15
        )

    raise ValueError(f"Unknown enhance mode: {mode}")
