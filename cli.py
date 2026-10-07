#!/usr/bin/env python3
"""
AI Document Scanner
--------------------
Detects every document-like region in a photo (handles multiple documents
in one frame, overlapping or at different angles), corrects perspective
and rotation for each one independently, cleans it up into a scan-style
image, and optionally runs OCR or bundles it into a PDF.

Usage:
    python cli.py photo.jpg
    python cli.py photo.jpg --ocr
    python cli.py photo.jpg --ocr --pdf -o results/

    # Process every image in a folder:
    python cli.py --batch photos/ --ocr --pdf -o results/
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

from scanner.segment import find_document_regions
from scanner.detect import find_document_contour
from scanner.transform import four_point_transform
from scanner.deskew import auto_deskew
from scanner.enhance import enhance_scan
from scanner.ocr import extract_text, ocr_confidence

LOW_CONFIDENCE_THRESHOLD = 40
from scanner.annotate import draw_regions

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Scan one or more photographed documents.")
    parser.add_argument("image", nargs="?", default=None,
                         help="Path to a single input image")
    parser.add_argument("--batch", metavar="FOLDER", default=None,
                         help="Process every image in FOLDER instead of a single file")
    parser.add_argument("-o", "--output", default="output", help="Output directory (default: output)")
    parser.add_argument("--mode", choices=["adaptive", "gray", "color"], default="adaptive",
                         help="Output style: adaptive B&W scan (default), gray, or color")
    parser.add_argument("--ocr", action="store_true", help="Extract text from each scan into a .txt file")
    parser.add_argument("--pdf", action="store_true", help="Also save each scan as a PDF")
    parser.add_argument("--lang", default="eng", help="OCR language code (default: eng)")
    parser.add_argument("--no-deskew", action="store_true",
                         help="Skip automatic 90/180/270 degree rotation correction")
    parser.add_argument("--min-area-ratio", type=float, default=0.01,
                         help="Minimum document size as a fraction of the photo's area (default: 0.01)")
    parser.add_argument("--max-area-ratio", type=float, default=0.92,
                         help="Maximum document size as a fraction of the photo's area (default: 0.92)")
    args = parser.parse_args()

    if not args.image and not args.batch:
        parser.error("provide an image path, or use --batch FOLDER")
    if args.image and args.batch:
        parser.error("use either a single image or --batch, not both")

    return args


def list_images(folder):
    if not os.path.isdir(folder):
        sys.exit(f"Error: folder not found: {folder}")
    files = [
        os.path.join(folder, f) for f in sorted(os.listdir(folder))
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]
    if not files:
        sys.exit(f"Error: no images found in {folder}")
    return files


def find_regions_with_fallback(image, args):
    """Multi-document segmentation first; if that finds nothing, fall back
    to the original single-largest-contour detector; if that also finds
    nothing, treat the whole photo as one document."""
    regions = find_document_regions(
        image, min_area_ratio=args.min_area_ratio, max_area_ratio=args.max_area_ratio
    )
    if regions:
        return regions

    single = find_document_contour(image)
    if single is not None:
        # cv2.contourArea requires float32/int32 - find_document_contour
        # returns float64 (points * ratio promotes the dtype), so cast
        # BEFORE computing area, not after, or this throws.
        points = single.astype("float32")
        return [{"points": points, "area": float(cv2.contourArea(points))}]

    h, w = image.shape[:2]
    full_frame = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    return [{"points": full_frame, "area": float(h * w)}]


def process_image(image_path, output_dir, args):
    """Run the full multi-document pipeline on one photo. Returns a
    result dict and raises on an unreadable file."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"could not read image (unsupported format?): {image_path}")

    basename = os.path.splitext(os.path.basename(image_path))[0]
    regions = find_regions_with_fallback(image, args)
    multi = len(regions) > 1

    documents = []
    for i, region in enumerate(regions, start=1):
        warped = four_point_transform(image, region["points"])

        rotation = 0
        if not args.no_deskew:
            warped, rotation = auto_deskew(warped)

        enhanced = enhance_scan(warped, mode=args.mode)

        suffix = f"_doc{i}" if multi else ""
        out_image_path = os.path.join(output_dir, f"{basename}{suffix}_scanned.jpg")
        cv2.imwrite(out_image_path, enhanced)

        doc_result = {
            "index": i,
            "rotation_applied": rotation,
            "scanned_image": out_image_path,
        }

        if args.ocr:
            text = extract_text(enhanced, lang=args.lang)
            out_text_path = os.path.join(output_dir, f"{basename}{suffix}.txt")
            with open(out_text_path, "w", encoding="utf-8") as f:
                f.write(text)
            doc_result["text_file"] = out_text_path

            confidence = ocr_confidence(enhanced, lang=args.lang)
            doc_result["ocr_confidence"] = confidence
            doc_result["low_confidence"] = confidence is None or confidence < LOW_CONFIDENCE_THRESHOLD

        if args.pdf:
            import img2pdf
            out_pdf_path = os.path.join(output_dir, f"{basename}{suffix}.pdf")
            with open(out_pdf_path, "wb") as f:
                f.write(img2pdf.convert(out_image_path))
            doc_result["pdf_file"] = out_pdf_path

        documents.append(doc_result)

    result = {"source": image_path, "documents_detected": len(regions), "documents": documents}

    if multi:
        overview = draw_regions(image, regions)
        overview_path = os.path.join(output_dir, f"{basename}_overview.jpg")
        cv2.imwrite(overview_path, overview)
        result["overview_image"] = overview_path

    report_path = os.path.join(output_dir, f"{basename}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    result["report"] = report_path

    return result


def main():
    args = parse_args()

    if args.pdf:
        try:
            import img2pdf  # noqa: F401
        except ImportError:
            sys.exit("img2pdf is required for --pdf. Install with: pip install img2pdf")

    os.makedirs(args.output, exist_ok=True)

    if args.batch:
        images = list_images(args.batch)
        print(f"Found {len(images)} image(s) in {args.batch}")

        results = []
        failed = []
        for i, path in enumerate(images, start=1):
            print(f"[{i}/{len(images)}] {os.path.basename(path)}")
            try:
                r = process_image(path, args.output, args)
                print(f"  {r['documents_detected']} document(s) found")
                results.append(r)
            except Exception as exc:
                print(f"  failed: {exc}")
                failed.append({"source": path, "error": str(exc)})

        total_docs = sum(r["documents_detected"] for r in results)
        low_confidence = [
            {"source": r["source"], "scanned_image": d["scanned_image"], "ocr_confidence": d["ocr_confidence"]}
            for r in results for d in r["documents"]
            if d.get("low_confidence")
        ]

        summary_path = os.path.join(args.output, "batch_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "images_processed": len(results),
                "images_failed": len(failed),
                "total_documents_detected": total_docs,
                "low_confidence_documents": low_confidence,
                "processed": results,
                "failed": failed,
            }, f, indent=2)

        print(f"\nDone. {len(results)} image(s) succeeded, {len(failed)} failed, "
              f"{total_docs} document(s) detected total.")
        if low_confidence and args.ocr:
            print(f"{len(low_confidence)} document(s) flagged as low-confidence OCR - worth a manual look:")
            for item in low_confidence:
                conf = item["ocr_confidence"]
                conf_str = f"{conf:.0f}" if conf is not None else "n/a"
                print(f"  [{conf_str}] {item['scanned_image']}")
        print(f"Summary -> {summary_path}")

    else:
        if not os.path.exists(args.image):
            sys.exit(f"Error: input file not found: {args.image}")

        print("Detecting document region(s)...")
        result = process_image(args.image, args.output, args)
        print(f"Found {result['documents_detected']} document(s).")
        for doc in result["documents"]:
            tag = f"doc {doc['index']}" if result["documents_detected"] > 1 else "scan"
            rot = f" (rotated {doc['rotation_applied']} deg)" if doc["rotation_applied"] else ""
            print(f"  {tag}{rot} -> {doc['scanned_image']}")
            if "text_file" in doc:
                print(f"    text -> {doc['text_file']}")
                if doc.get("low_confidence"):
                    conf = doc["ocr_confidence"]
                    conf_str = f"{conf:.0f}" if conf is not None else "n/a"
                    print(f"    WARNING: low OCR confidence ({conf_str}) - result may be unreliable")
            if "pdf_file" in doc:
                print(f"    pdf  -> {doc['pdf_file']}")
        if "overview_image" in result:
            print(f"Overview -> {result['overview_image']}")
        print(f"Report -> {result['report']}")
        print("Done.")


if __name__ == "__main__":
    main()
