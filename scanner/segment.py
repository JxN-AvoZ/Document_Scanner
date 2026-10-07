"""Detect MULTIPLE document-like regions in a single photo.

This differs from the original single-document approach (which just grabs
the single largest 4-point contour) by searching every contour in the
frame for plausible rectangular document shapes, so a photo containing
several documents - overlapping, at different angles, different sizes -
gets each one found separately rather than only the biggest or none.
"""
import cv2
import numpy as np


def find_document_regions(image, min_area_ratio=0.01, max_area_ratio=0.92,
                           min_solidity=0.85, iou_threshold=0.3):
    """Find candidate document regions in `image`.

    Args:
        min_area_ratio / max_area_ratio: candidate area as a fraction of
            the full image area (filters out noise specks and the
            occasional contour that wraps the whole frame)
        min_solidity: contour_area / convex_hull_area - real documents are
            solid rectangles, so a low solidity usually means the contour
            is a messy, non-rectangular blob (shadow, clutter, etc.)
        iou_threshold: candidates overlapping more than this are treated
            as duplicate detections of the same physical document (common
            since Canny often produces both an inner and outer edge for
            the same object) - only the larger one is kept

    Returns:
        List of dicts: {"points": 4x2 float32 corner array (as given by
        minAreaRect, NOT yet corner-ordered), "area": float}, largest first.
    """
    h, w = image.shape[:2]
    img_area = h * w
    min_area = img_area * min_area_ratio
    max_area = img_area * max_area_ratio

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < min_solidity:
            continue

        box = _quad_corners(c)
        candidates.append({"points": box, "area": float(area)})

    candidates.sort(key=lambda d: d["area"], reverse=True)
    return _suppress_overlapping(candidates, iou_threshold)


def _quad_corners(contour):
    """Get the 4 corner points of `contour`, preferring a direct polygon
    approximation (which preserves real perspective/keystone distortion -
    a photographed page's true shape is a trapezoid, not a rectangle)
    over minAreaRect's rotated-but-unskewed bounding box, which is only
    used as a fallback for contours that don't reduce cleanly to a quad.
    """
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    for epsilon_factor in (0.02, 0.04, 0.06):
        approx = cv2.approxPolyDP(hull, epsilon_factor * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")

    rect = cv2.minAreaRect(contour)
    return cv2.boxPoints(rect).astype("float32")


def _polygon_iou(poly_a, poly_b):
    try:
        inter_area, _ = cv2.intersectConvexConvex(poly_a.astype("float32"), poly_b.astype("float32"))
    except cv2.error:
        return 0.0
    area_a = cv2.contourArea(poly_a)
    area_b = cv2.contourArea(poly_b)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _suppress_overlapping(candidates, iou_threshold):
    kept = []
    for cand in candidates:
        if all(_polygon_iou(cand["points"], k["points"]) < iou_threshold for k in kept):
            kept.append(cand)
    return kept
