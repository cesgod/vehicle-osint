"""Image pipeline: OpenCV preprocessing -> Google Cloud Vision OCR (or local EasyOCR) -> plate extraction.

Engines, in order of preference:
  1. google-vision  Advanced mode (signed in). TEXT_DETECTION + OBJECT_LOCALIZATION.
  2. easyocr        Optional offline fallback, used when installed and not signed in.
  3. none           Preprocessing still runs (plate-region crop saved) but no text is read.

QUOTA (Cloud Vision)
--------------------
Billing is per feature per image: one image here = 2 units (text + objects). The first
1,000 units/month of each feature are free; after that roughly USD 1.50 per 1,000.
Default request limit is 1,800/minute. This module:
  * downsizes images to 1600 px before upload (smaller payload, same OCR quality),
  * counts units locally and stops at VISION_MONTHLY_BUDGET,
  * never re-sends an image when no plate is found.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz

from . import config, google_auth
from .analyzer import coerce_plate, normalize_plate
from .models import ImageScan, Result, Search, add_usage, usage

VEHICLE_LABELS = {"car", "vehicle", "truck", "motorcycle", "bus", "van", "land vehicle", "license plate",
                  "vehicle registration plate", "pickup truck", "taxi"}
MAX_SIDE = 1600
_easyocr_reader = None


# --------------------------------------------------------------------------- #
# OpenCV preprocessing
# --------------------------------------------------------------------------- #
def load_image(path: Path) -> np.ndarray | None:
    """Read via numpy so non-ASCII Windows paths work."""
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def _resize(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img


def _skew_angle(gray: np.ndarray) -> float:
    """Median angle of near-horizontal edges, in degrees."""
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=gray.shape[1] // 6, maxLineGap=12)
    if lines is None:
        return 0.0
    angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in lines[:, 0]]
    angles = [a for a in angles if abs(a) < 20]
    return float(np.median(angles)) if len(angles) >= 3 else 0.0


def preprocess(img: np.ndarray) -> np.ndarray:
    """Resize, deskew and boost local contrast (CLAHE on the lightness channel)."""
    img = _resize(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angle = _skew_angle(gray)
    if 0.7 < abs(angle) < 20:
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def plate_regions(img: np.ndarray, limit: int = 4) -> list[tuple[int, int, int, int]]:
    """Plate-shaped rectangles (x, y, w, h): 4-sided contours with a plate-like aspect ratio."""
    gray = cv2.bilateralFilter(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 9, 60, 60)
    edges = cv2.dilate(cv2.Canny(gray, 50, 180), np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area_img = img.shape[0] * img.shape[1]
    boxes = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:60]:
        approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        if len(approx) == 4 and h > 0 and 2.0 <= w / h <= 6.0 and 0.002 <= (w * h) / area_img <= 0.35:
            boxes.append((x, y, w, h))
    return boxes[:limit]


def _crop(img: np.ndarray, box: tuple[int, int, int, int], pad: float = 0.15) -> np.ndarray:
    x, y, w, h = box
    px, py = int(w * pad), int(h * pad)
    return img[max(0, y - py): y + h + py, max(0, x - px): x + w + px]


def dominant_color(img: np.ndarray) -> str:
    """Name of the dominant colour in a region (k-means in BGR, named via HSV)."""
    h, w = img.shape[:2]
    core = img[h // 5: h - h // 5, w // 5: w - w // 5]
    if core.size == 0:
        return ""
    pixels = cv2.resize(core, (48, 48), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
    _, labels, centers = cv2.kmeans(pixels, 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0),
                                    3, cv2.KMEANS_PP_CENTERS)
    bgr = centers[np.bincount(labels.ravel()).argmax()].astype(np.uint8).reshape(1, 1, 3)
    hue, sat, val = (int(v) for v in cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0])
    if val < 55:
        return "negro"
    if sat < 38:
        return "blanco" if val > 195 else "plata" if val > 130 else "gris"
    for limit, name in ((8, "rojo"), (20, "naranja"), (34, "amarillo"), (85, "verde"), (100, "celeste"),
                        (130, "azul"), (160, "violeta"), (180, "rojo")):
        if hue <= limit:
            return "marrón" if name in ("naranja", "rojo") and val < 130 else name
    return ""


# --------------------------------------------------------------------------- #
# OCR engines -> list of (text, confidence, box)
# --------------------------------------------------------------------------- #
def easyocr_available() -> bool:
    try:
        import easyocr  # noqa: F401
        return True
    except Exception:
        return False


def vision_month() -> str:
    return datetime.now().strftime("%Y-%m")


def _ocr_vision(db, img: np.ndarray) -> tuple[list[tuple[str, float, tuple]], str, list[dict]]:
    from google.cloud import vision

    if usage(db, "vision", vision_month()) + 2 > config.vision_monthly_budget():
        raise RuntimeError(f"Local monthly Vision budget of {config.vision_monthly_budget()} units reached "
                           "(VISION_MONTHLY_BUDGET).")
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Could not encode image.")
    request = vision.AnnotateImageRequest(
        image=vision.Image(content=buf.tobytes()),
        features=[vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION),
                  vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION, max_results=10)],
    )
    response = google_auth.vision_client().batch_annotate_images(requests=[request]).responses[0]
    add_usage(db, "vision", vision_month(), 2)
    if response.error.message:
        raise RuntimeError(f"Cloud Vision: {response.error.message}")

    words = []
    annotations = list(response.text_annotations)
    full_text = annotations[0].description if annotations else ""
    for a in annotations[1:]:
        xs, ys = [v.x for v in a.bounding_poly.vertices], [v.y for v in a.bounding_poly.vertices]
        words.append((a.description, 0.95, (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))))

    h, w = img.shape[:2]
    vehicles = []
    for o in response.localized_object_annotations:
        if o.name.lower() in VEHICLE_LABELS:
            xs = [v.x * w for v in o.bounding_poly.normalized_vertices]
            ys = [v.y * h for v in o.bounding_poly.normalized_vertices]
            vehicles.append({"name": o.name, "score": round(o.score, 2),
                             "box": [int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys))]})
    return words, full_text, vehicles


def _ocr_easyocr(img: np.ndarray) -> tuple[list[tuple[str, float, tuple]], str]:
    global _easyocr_reader
    import easyocr

    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)
    words = []
    for box, text, conf in _easyocr_reader.readtext(img):
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        for token in text.split() or [text]:
            words.append((token, float(conf), (int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys)))))
    return words, " ".join(w[0] for w in words)


def _union(a: tuple, b: tuple) -> tuple:
    x, y = min(a[0], b[0]), min(a[1], b[1])
    return x, y, max(a[0] + a[2], b[0] + b[2]) - x, max(a[1] + a[3], b[1] + b[3]) - y


def plates_from_words(words: list[tuple[str, float, tuple]], target: str = "") -> list[dict]:
    """Find plate-shaped tokens (single words, or two adjacent words such as 'ABCD' '123').

    Confidence = OCR confidence, minus 0.08 per character that needed a 0/O-style correction;
    raised to 0.99 when the text equals the target plate.
    """
    target = normalize_plate(target)
    candidates = [(normalize_plate(t), c, b) for t, c, b in words if normalize_plate(t)]
    pairs = [(a[0] + b[0], min(a[1], b[1]), _union(a[2], b[2])) for a, b in zip(candidates, candidates[1:])]
    found: dict[str, dict] = {}
    for token, conf, box in candidates + pairs:
        fit = coerce_plate(token)
        if not fit:
            continue
        plate, fmt, fixes = fit
        score = max(0.3, conf - 0.08 * fixes)
        if target:
            score = 0.99 if plate == target else score * (0.6 + 0.4 * fuzz.ratio(plate, target) / 100)
        if plate not in found or score > found[plate]["confidence"]:
            found[plate] = {"plate": plate, "format": fmt, "confidence": round(score, 2), "corrected": fixes > 0,
                            "read_as": token, "box": [int(v) for v in box]}
    return sorted(found.values(), key=lambda p: -p["confidence"])


# --------------------------------------------------------------------------- #
# Cross-reference against stored search results
# --------------------------------------------------------------------------- #
def cross_reference(db, plates: list[str]) -> list[dict]:
    """Where else has this plate been seen? Checks past searches and every stored result."""
    matches = []
    if not plates:
        return matches
    for s in db.query(Search).filter(Search.plate != "").all():
        for p in plates:
            score = fuzz.ratio(p, s.plate)
            if score >= 85:
                matches.append({"kind": "search", "search_id": s.id, "plate": s.plate,
                                "match": "exact" if score == 100 else "fuzzy", "score": round(score)})
    for r in db.query(Result).all():
        hit = next((p for p in plates if p in (r.plates_found or [])), None)
        if hit:
            matches.append({"kind": "result", "result_id": r.id, "search_id": r.search_id, "plate": hit,
                            "match": "exact", "score": 100, "title": r.title, "url": r.url})
    return matches[:50]


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def analyze_image(db, path: Path, target_plate: str = "", original_name: str = "") -> ImageScan:
    scan = ImageScan(filename=original_name or path.name, source_path=str(path))
    raw = load_image(path)
    if raw is None:
        scan.error = "Not a readable image."
        db.add(scan); db.commit()
        return scan

    img = preprocess(raw)
    words: list = []
    vehicles: list[dict] = []
    try:
        if google_auth.get_credentials() is not None:
            scan.engine = "google-vision"
            words, scan.raw_text, vehicles = _ocr_vision(db, img)
        elif easyocr_available():
            scan.engine = "easyocr"
            words, scan.raw_text = _ocr_easyocr(img)
            if not plates_from_words(words, target_plate):  # second pass on plate-shaped crops
                for box in plate_regions(img):
                    crop = cv2.resize(_crop(img, box), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    more, _ = _ocr_easyocr(crop)
                    words += [(t, c, box) for t, c, _b in more]
        else:
            scan.error = "No OCR engine: sign in with Google (Advanced mode) or install the optional EasyOCR fallback."
    except Exception as exc:
        scan.error = str(exc)[:500]

    plates = plates_from_words(words, target_plate)
    scan.plates = plates
    scan.vehicles = vehicles
    if plates:
        scan.best_plate, scan.confidence = plates[0]["plate"], plates[0]["confidence"]

    # Evidence crop: the plate if read, else the best plate-shaped region.
    crop_box = tuple(plates[0]["box"]) if plates else next(iter(plate_regions(img)), None)
    if crop_box and crop_box[2] > 4 and crop_box[3] > 4:
        crop = _crop(img, crop_box, pad=0.35)
        if crop.size:
            scan.crop_file = f"{uuid.uuid4().hex}.jpg"
            cv2.imwrite(str(config.CROPS_DIR / scan.crop_file), crop)

    # Dominant colour of each detected vehicle (or the whole frame when none was localized).
    regions = [_crop(img, tuple(v["box"]), pad=0) for v in vehicles if v["name"].lower() not in ("license plate", "vehicle registration plate")] or [img]
    colors = []
    for region in regions[:3]:
        name = dominant_color(region) if region.size else ""
        if name and name not in colors:
            colors.append(name)
    scan.colors = colors

    scan.matches = cross_reference(db, [p["plate"] for p in plates])
    db.add(scan); db.commit()
    return scan


def list_folder_images(folder: str, limit: int = 200) -> list[Path]:
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS]
    return files[:limit]
