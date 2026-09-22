"""
License-plate finder + reader for the WAY-O-AI school prototype.

Steps:
1. Clean the photo and find edges.
2. Keep rectangle-shaped regions that look like number plates.
3. Read letters/digits with Tesseract OCR.
4. Fix common 0/O and 1/I mistakes, then snap to a known plate if very close.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytesseract


TESSERACT_CONFIG = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
REGISTRY_FILE = Path(__file__).parent / "data" / "vehicles.csv"


@dataclass
class PlateHit:
    text: str
    box: tuple[int, int, int, int]
    crop: np.ndarray
    score: int = 0


def _registry_plates() -> list[str]:
    if not REGISTRY_FILE.exists():
        return []
    df = pd.read_csv(REGISTRY_FILE)
    return ["".join(ch for ch in str(p).upper() if ch.isalnum()) for p in df["plate"]]


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins, delete, sub = cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def _clean(raw: str) -> str:
    return "".join(ch for ch in raw.upper() if ch.isalnum())


def _swap_confusions(text: str) -> list[str]:
    """Try both letter and digit readings for look-alike characters."""
    variants = {text}
    table_to_digit = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8"})
    table_to_letter = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"})
    variants.add(text.translate(table_to_digit))
    variants.add(text.translate(table_to_letter))
    # Mixed: first two chars letters, next two digits (MH02...), last four digits
    mixed = list(text)
    if len(mixed) >= 8:
        for i, ch in enumerate(mixed):
            if i < 2 and ch.isdigit():
                mixed[i] = {"0": "O", "1": "I", "5": "S", "8": "B"}.get(ch, ch)
            elif 2 <= i <= 3 and ch.isalpha():
                mixed[i] = {"O": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "B": "8"}.get(ch, ch)
            elif i >= len(mixed) - 4 and ch.isalpha():
                mixed[i] = {"O": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2"}.get(ch, ch)
        variants.add("".join(mixed))
    return list(variants)


def snap_to_registry(text: str) -> tuple[str, int]:
    """Return (best_text, distance). Distance 0 means exact known plate."""
    text = _clean(text)
    plates = _registry_plates()
    candidates = _swap_confusions(text) + [text]
    best_text, best_d = text, 99
    for cand in candidates:
        if cand in plates:
            return cand, 0
        for plate in plates:
            d = _levenshtein(cand, plate)
            if d < best_d:
                best_d, best_text = d, plate if d <= 2 else cand
    if best_d <= 2:
        return best_text, best_d
    return text, best_d


def _looks_like_plate(text: str) -> bool:
    if len(text) < 6 or len(text) > 12:
        return False
    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    return letters >= 2 and digits >= 3


def _resize_max(image: np.ndarray, max_width: int = 900) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / float(w)
    return cv2.resize(image, (int(w * scale), int(h * scale)))


def _ocr_plate(crop: np.ndarray) -> str:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.resize(gray, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 7, 40, 40)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    best = ""
    for img in (thresh, cv2.bitwise_not(thresh)):
        raw = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
        text = _clean(raw)
        if _looks_like_plate(text) and len(text) >= len(best):
            best = text
        elif not best and len(text) >= 6:
            best = text
    return best


def find_plate_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, 30, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    h_img, w_img = image.shape[:2]
    min_area = (w_img * h_img) * 0.004
    max_area = (w_img * h_img) * 0.35
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        area, ratio = w * h, w / float(h)
        if area < min_area or area > max_area:
            continue
        if ratio < 1.8 or ratio > 7.5:
            continue
        if w < 80 or h < 18:
            continue
        boxes.append((x, y, w, h))
    return boxes


def detect_plates(image: np.ndarray, max_hits: int = 1) -> list[PlateHit]:
    work = _resize_max(image)
    ranked: list[PlateHit] = []

    def consider(crop: np.ndarray, box: tuple[int, int, int, int]) -> None:
        if crop.size == 0:
            return
        raw = _ocr_plate(crop)
        if not raw:
            return
        snapped, distance = snap_to_registry(raw)
        if not _looks_like_plate(snapped) and distance > 2:
            return
        score = 100 - distance * 20 + min(len(snapped), 10)
        ranked.append(PlateHit(text=snapped, box=box, crop=crop, score=score))

    for (x, y, w, h) in find_plate_boxes(work):
        pad_x, pad_y = int(w * 0.04), int(h * 0.12)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1 = min(work.shape[1], x + w + pad_x)
        y1 = min(work.shape[0], y + h + pad_y)
        consider(work[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0))

    if not ranked:
        h, w = work.shape[:2]
        band = work[int(h * 0.45) : int(h * 0.95), int(w * 0.08) : int(w * 0.92)]
        consider(band, (int(w * 0.08), int(h * 0.45), int(w * 0.84), int(h * 0.50)))

    ranked.sort(key=lambda hit: hit.score, reverse=True)
    unique: list[PlateHit] = []
    seen = set()
    for hit in ranked:
        if hit.text in seen:
            continue
        seen.add(hit.text)
        unique.append(hit)
        if len(unique) >= max_hits:
            break
    return unique


def annotate(image: np.ndarray, hits: list[PlateHit], extra_lines: list[str] | None = None) -> np.ndarray:
    canvas = _resize_max(image).copy()
    for hit in hits:
        x, y, w, h = hit.box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 220, 80), 2)
        cv2.putText(canvas, hit.text, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 80), 2)
    if extra_lines:
        y = 28
        for line in extra_lines:
            cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3)
            cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
            y += 22
    return canvas
