#!/usr/bin/env python3
import argparse
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# ---------- geometry helpers ----------
def _order_points(pts: np.ndarray) -> np.ndarray:
    # pts: (4,2) float
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]        # top-left
    rect[2] = pts[np.argmax(s)]        # bottom-right
    rect[1] = pts[np.argmin(diff)]     # top-right
    rect[3] = pts[np.argmax(diff)]     # bottom-left
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    # compute new image width/height
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array(
        [[0, 0],
         [maxWidth - 1, 0],
         [maxWidth - 1, maxHeight - 1],
         [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# ---------- document detection ----------
def find_document_contour(frame: np.ndarray) -> Optional[np.ndarray]:
    # returns 4-point contour (float32) if found
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gray, 50, 150)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 0.15 * frame.shape[0] * frame.shape[1]:  # big enough
                return approx.reshape(4, 2).astype("float32")
    return None

def clean_scan(img: np.ndarray) -> np.ndarray:
    # B/W "scanner" style using adaptive threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # increase contrast a bit
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15
    )
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

# ---------- main loop ----------
def to_pil_rgb(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def save_pdf(pages_bgr: List[np.ndarray], out_path: str):
    pil_pages = [to_pil_rgb(p) for p in pages_bgr]
    # Convert all to RGB just in case
    pil_pages = [p.convert("RGB") for p in pil_pages]
    pil_pages[0].save(out_path, save_all=True, append_images=pil_pages[1:], format="PDF")

def draw_hud(frame: np.ndarray, quad: Optional[np.ndarray], page_count: int):
    hud = frame.copy()
    h, w = hud.shape[:2]
    # dark border for readability
    cv2.rectangle(hud, (0, 0), (w - 1, h - 1), (0, 0, 0), 40)
    alpha = 0.15
    frame[:] = cv2.addWeighted(hud, alpha, frame, 1 - alpha, 0)

    if quad is not None:
        pts = quad.astype(int).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
        for i, (x, y) in enumerate(quad.astype(int)):
            cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(frame, str(i+1), (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    help_text = [
        "CamScanner",
        "[c] capture page   [u] undo   [s] save PDF   [r] toggle clean   [q/ESC] quit",
        f"Pages captured: {page_count}",
        "Tip: Hold the document flat; keep it well lit for best edges.",
    ]
    y0 = 24
    for line in help_text:
        cv2.putText(frame, line, (16, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y0 += 26

def main():
    ap = argparse.ArgumentParser(description="Camera-based document scanner")
    ap.add_argument("--cam", type=int, default=0, help="Camera index (default 0)")
    ap.add_argument("--out", type=str, default="scan.pdf", help="Output PDF file")
    ap.add_argument("--width", type=int, default=1280, help="Capture width request")
    ap.add_argument("--height", type=int, default=720, help="Capture height request")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("ERROR: Cannot open camera", file=sys.stderr)
        sys.exit(1)

    # Try to set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cv2.namedWindow("Scanner", cv2.WINDOW_NORMAL)

    pages: List[np.ndarray] = []
    use_clean = True

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: Failed to read from camera", file=sys.stderr)
            break

        quad = find_document_contour(frame)
        overlay = frame.copy()
        draw_hud(overlay, quad, len(pages))
        cv2.imshow("Scanner", overlay)
        key = cv2.waitKey(10) & 0xFF

        if key in (27, ord('q')):  # ESC or q
            break
        elif key == ord('r'):
            use_clean = not use_clean
        elif key == ord('u'):
            if pages:
                pages.pop()
        elif key == ord('c'):
            # capture current frame
            snap = frame.copy()
            if quad is not None:
                warped = four_point_transform(snap, quad)
            else:
                warped = snap
            if use_clean:
                warped = clean_scan(warped)
            # Optional: upscale small scans for nicer PDF rendering
            h, w = warped.shape[:2]
            if max(h, w) < 1200:
                scale = 1200 / max(h, w)
                warped = cv2.resize(warped, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
            pages.append(warped)
            # flash feedback
            flash = np.full_like(overlay, 255)
            cv2.addWeighted(flash, 0.4, overlay, 0.6, 0, overlay)
            cv2.imshow("Scanner", overlay)
            cv2.waitKey(80)
        elif key == ord('s'):
            if pages:
                try:
                    save_pdf(pages, args.out)
                    print(f"Saved PDF: {args.out}")
                except Exception as e:
                    print(f"ERROR: could not save PDF: {e}", file=sys.stderr)
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

