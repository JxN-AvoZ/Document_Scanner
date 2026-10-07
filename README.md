# AI Document Scanner

A command-line tool that finds every document in a photo — even multiple
documents in one frame, overlapping or at different angles — and turns
each one into a clean, flat, correctly-oriented scan, with optional OCR
text extraction and PDF export.

For each photo, it will:

1. Detect every document-like region in the frame (not just the single
   biggest one)
2. Correct the perspective of each one so it looks flat and top-down
3. Auto-correct 90/180/270° rotation per document, independently
4. Clean it up into a crisp black-and-white "scan" look
5. *(optional)* Extract the text with OCR
6. *(optional)* Save the result as a PDF

A single photo with one document is handled the same way it always was —
multi-document handling only kicks in when there's more than one to find.

## Example

Run it on the included sample:

```bash
python cli.py examples/sample_photo.jpg --ocr --pdf -o examples/output
```

```
Detecting document region(s)...
Found 1 document(s).
  scan -> examples/output/sample_photo_scanned.jpg
    text -> examples/output/sample_photo.txt
    pdf  -> examples/output/sample_photo.pdf
Report -> examples/output/sample_photo_report.json
Done.
```

Point it at a photo with several documents in frame and it finds each one:

```
Detecting document region(s)...
Found 3 document(s).
  doc 1 -> output/photo_doc1_scanned.jpg
  doc 2 -> output/photo_doc2_scanned.jpg
  doc 3 (rotated 90 deg) -> output/photo_doc3_scanned.jpg
Overview -> output/photo_overview.jpg
Report -> output/photo_report.json
```

The `_overview.jpg` file shows every detected document outlined and
numbered on the original photo — useful for sanity-checking detection
before trusting the output.

## Installation

```bash
git clone https://github.com/<your-username>/document-scanner.git
cd document-scanner
pip install -r requirements.txt
```

OCR requires the Tesseract binary to be installed separately (it's a
separate engine, not a Python package, and rotation detection depends on
it too):

- **macOS:** `brew install tesseract`
- **Ubuntu/Debian:** `sudo apt install tesseract-ocr`
- **Windows:** [installer here](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

```bash
python cli.py <image> [options]
```

| Flag | Description |
|---|---|
| `-o`, `--output` | Output directory (default: `output/`) |
| `--mode` | `adaptive` (default, B&W scan look), `gray`, or `color` |
| `--ocr` | Extract text into a `.txt` file per document |
| `--pdf` | Also save each document as a `.pdf` |
| `--lang` | OCR language code, e.g. `eng`, `spa` (default: `eng`) |
| `--no-deskew` | Skip automatic 90/180/270° rotation correction |
| `--min-area-ratio` | Smallest a document can be, as a fraction of the photo (default: 0.01) |
| `--max-area-ratio` | Largest a document can be, as a fraction of the photo (default: 0.92) |

When `--ocr` is used, every document also gets an OCR confidence score
(Tesseract's own mean word-level confidence, 0-100). Anything below 40 is
flagged `low_confidence` in the report — a cheap way to catch garbage
output (wrong rotation, no real document found, blank/noisy image)
without reading every `.txt` file by hand. In batch mode, flagged
documents are listed at the end of the run so you can triage a large
dataset quickly:

```
3 document(s) flagged as low-confidence OCR - worth a manual look:
  [29] results/pile_doc2_scanned.jpg
  [12] results/weird_angle_scanned.jpg
  [35] results/blurry_scanned.jpg
```

A low score doesn't mean the text is wrong on a word-by-word basis - just
that Tesseract itself wasn't confident, which in practice correlates well
with genuinely bad input (unrecoverable rotation, overlapping documents
with no clean boundary, no document actually found).

### Examples

```bash
# Basic scan
python cli.py receipt.jpg

# Scan + extract text
python cli.py receipt.jpg --ocr

# Scan + text + PDF, custom output folder
python cli.py receipt.jpg --ocr --pdf -o results/

# A photo of several small receipts side by side
python cli.py receipts_pile.jpg --ocr --min-area-ratio 0.005
```

### Batch mode

Point it at a folder instead of a single file to process every image in it
— handy for running a whole test set at once:

```bash
python cli.py --batch photos/ --ocr --pdf -o results/
```

A file that fails to process (unreadable, wrong format, etc.) is skipped
with an error rather than stopping the whole batch. A `batch_summary.json`
is written to the output folder with a total document count and the full
per-image breakdown:

```json
{
  "images_processed": 12,
  "images_failed": 1,
  "total_documents_detected": 15,
  "processed": [
    {"source": "photos/pile.jpg", "documents_detected": 3,
     "documents": [{"index": 1, "rotation_applied": 0, "scanned_image": "...", "text_file": "..."}],
     "overview_image": "results/pile_overview.jpg", "report": "results/pile_report.json"}
  ],
  "failed": [
    {"source": "photos/corrupt.jpg", "error": "could not read image (unsupported format?): photos/corrupt.jpg"}
  ]
}
```

## How it works

- **Segmentation (`scanner/segment.py`):** grayscale → blur → Canny edges →
  every contour in the frame is checked (not just the largest), filtered by
  size and solidity (`area / convex_hull_area`, to reject non-rectangular
  blobs), and overlapping duplicate detections are suppressed by IoU. Each
  surviving contour's corners are found via polygon approximation, which —
  unlike a rotated bounding box — preserves the true keystone/trapezoid
  shape a document has when photographed at an angle, rather than forcing
  it into an unskewed rectangle.
- **Fallback (`scanner/detect.py`):** if segmentation finds nothing (e.g. a
  single document filling almost the whole frame), the original
  single-largest-contour detector runs instead; if that also finds
  nothing, the whole photo is used as-is.
- **Perspective correction (`scanner/transform.py`):** a 4-point transform
  warps each detected region into a flat rectangle.
- **Rotation correction (`scanner/deskew.py`):** Tesseract's orientation
  detection (OSD) runs on each already-isolated, already-flattened crop —
  not on the full multi-document photo, where OSD gets confused by several
  different text orientations at once — and rotates it to upright if
  confident. A crop with too little text for OSD to judge is retried once
  upscaled before giving up and leaving it unrotated.
- **Enhancement (`scanner/enhance.py`):** adaptive thresholding gives the
  classic scanned-paper look.
- **OCR (`scanner/ocr.py`):** Tesseract via `pytesseract`, using `--psm 6`
  (treat the crop as one uniform text block) rather than Tesseract's
  default automatic page-layout mode, which tends to misfire on a single
  cropped scan.
- **Overview annotation (`scanner/annotate.py`):** draws every detected
  region's outline and index on the original photo, only produced when
  more than one document is found.

## Project structure

```
document-scanner/
├── cli.py              # entry point
├── scanner/
│   ├── segment.py        # find every document region in the frame
│   ├── detect.py         # single-document fallback (original approach)
│   ├── transform.py      # perspective correction
│   ├── deskew.py          # per-document rotation correction
│   ├── enhance.py        # scan-style cleanup
│   ├── ocr.py             # text extraction
│   └── annotate.py        # overview image for multi-document photos
├── examples/              # sample input/output
└── requirements.txt
```

## Notes / limitations

This uses classic computer vision (edge + contour detection), not a
trained model, so it works best with reasonable contrast between each
document and the background/other documents:

- Documents that overlap each other, or touch with no visible gap, can be
  merged into a single detection or missed
- Very low-contrast photos (document blending into the background) may be
  missed entirely
- Rotation correction snaps to the nearest 90° and depends on Tesseract
  recognizing enough text to judge orientation confidently — a document
  that's skewed by some odd angle rather than a clean 90/180/270° will
  still get flattened by the perspective step, but may not end up
  perfectly upright
- A photo where several documents are stacked with significant overlap
  (rather than laid out separately) is a genuinely hard case for any
  contour-based approach and may need the documents re-photographed apart

**No handwriting recognition.** OCR is powered by Tesseract, which is
built and trained for printed/typed text. It will attempt to read
handwritten notes and usually return low-confidence nonsense rather than
accurate text — the `low_confidence` flag described above will generally
catch this, but don't expect handwritten pages to work even when it
doesn't trigger that flag.

**English by default, not auto-detected.** OCR language is fixed per run
via `--lang` (default `eng`) — it is never guessed from the image. Running
without the right `--lang` on a non-English document will produce garbled
text that can look like a bug but is just the wrong language model being
used. See [Usage](#usage) above for installing and combining language
packs.

**Photograph documents straight-on when possible.** The perspective
correction step can flatten a reasonable amount of tilt, but the whole
pipeline — region detection, corner finding, and the deskew step after it
— is built around the expectation that it's looking at a document shot
from roughly in front of it, not from a steep side angle or a sharp
overhead angle onto an angled surface. Photos taken at extreme angles are
more likely to fail segmentation entirely (falling back to whole-image
processing) or to produce a crop whose corners are slightly off, which
degrades OCR accuracy even when a document *is* detected. For best
results: shoot from directly in front of the document, fill as much of
the frame as reasonably possible, and keep a visible contrast between the
document and the surface it's on.

## License

MIT
