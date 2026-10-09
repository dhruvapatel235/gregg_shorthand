"""Turn a scanned page (image or PDF) into lines of word images."""
from pathlib import Path

import cv2
import numpy as np

from src.dataset import load_gray


def load_pages(path, dpi=200):
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        try:
            import pymupdf as fitz  # PyMuPDF (new name)
        except ImportError:
            import fitz
        pages = []
        for p in fitz.open(str(path)):
            pix = p.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
            pages.append(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width))
        return pages
    return [load_gray(path)]


def binarize(gray):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink


def _runs(mask, min_len, max_gap):
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(mask)])
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= max_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [r for r in merged if r[1] - r[0] >= min_len]


def segment_page(gray, word_gap=0.9, min_line_h=12):
    """
    Returns (lines, boxes). lines: list of lists of ink-on-dark word crops (uint8).
    boxes: list of (x, y, w, h) in reading order for the debug picture.
    word_gap: horizontal merge distance as a fraction of line height (raise it if one word is
    split into pieces, lower it if neighbouring words are glued together).
    """
    ink = binarize(gray)
    H, W = ink.shape
    row_mask = ink.sum(axis=1) / 255 > max(1, 0.004 * W)
    line_runs = _runs(row_mask, min_line_h, max_gap=max(3, int(0.012 * H)))
    lines, boxes = [], []
    for y0, y1 in line_runs:
        strip = ink[y0:y1]
        lh = y1 - y0
        k = max(5, int(word_gap * lh))
        dil = cv2.dilate(strip, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 3)))
        cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bxs = sorted(cv2.boundingRect(c) for c in cnts)
        words = []
        for x, y, w, h in bxs:
            if w * h < 0.01 * lh * lh or w < 4:
                continue
            pad = 3
            xa, xb = max(0, x - pad), min(W, x + w + pad)
            ya, yb = max(0, y0 + y - pad), min(H, y0 + y + h + pad)
            words.append(ink[ya:yb, xa:xb])
            boxes.append((xa, ya, xb - xa, yb - ya))
        if words:
            lines.append(words)
    return lines, boxes


def draw_boxes(gray, boxes, out_path):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(vis, str(i + 1), (x, max(10, y - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    cv2.imwrite(str(out_path), vis)
