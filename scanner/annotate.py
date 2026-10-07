"""Draw detected document region outlines on the original photo, numbered
to match the per-document output files - useful for sanity-checking
segmentation on a new photo before trusting the OCR output."""
import cv2
import numpy as np


def draw_regions(image, regions):
    annotated = image.copy()

    for i, r in enumerate(regions, start=1):
        pts = r["points"].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 200, 0), thickness=3)

        label_pos = tuple(r["points"][0].astype(int))
        cv2.putText(annotated, str(i), (label_pos[0] + 5, label_pos[1] + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)

    label = f"Documents found: {len(regions)}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(annotated, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
    cv2.putText(annotated, label, (10, th + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return annotated
