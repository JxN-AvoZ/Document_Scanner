"""Detect the four-corner outline of a document within a photo."""
import cv2


def find_document_contour(image, resize_height=500):
    """Return the 4 corner points of the largest document-like shape found
    in `image`, scaled back to the original image size. Returns None if no
    quadrilateral is found."""
    ratio = image.shape[0] / float(resize_height)
    resized = cv2.resize(image, (int(image.shape[1] / ratio), resize_height))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2) * ratio

    return None
