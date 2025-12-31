import os
import cv2
import numpy as np
from math import isfinite

# ---------- helpers ----------
def ensure_binary_white_ink(img_gray):
    """Ensure binary image with ink=white (255) on black background."""
    _, b = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(b == 255) > np.sum(b == 0):  # invert if background is white
        b = 255 - b
    return b

def hybrid_px_per_mm(img_gray, ink_mask, text_rect):
    """Estimate px/mm from hybrid cues (page + text width)."""
    H, W = img_gray.shape[:2]
    x, y, w, h = text_rect
    candidates = []

    if H > 600:
        candidates.append(H / 297.0)  # height of A4
    if W > 400:
        candidates.append(W / 210.0)  # width of A4
    if w > 200:
        candidates.append(w / 150.0)  # typical writing width

    candidates = [c for c in candidates if isfinite(c) and c > 0]
    if not candidates:
        px_per_mm = 6.0
    else:
        px_per_mm = float(np.median(candidates))

    return float(np.clip(px_per_mm, 4.0, 12.0))

def find_lines(ink_mask):
    """Detect handwriting lines via horizontal projection."""
    proj = np.sum(ink_mask > 0, axis=1).astype(np.float32)
    if proj.max() == 0:
        return []

    thr = max(10.0, proj.max() * 0.10)
    lines, in_line, start = [], False, 0

    for i, v in enumerate(proj):
        if v > thr and not in_line:
            start = i
            in_line = True
        elif v <= thr and in_line:
            end = i
            if end - start > 10:
                lines.append((start, end))
            in_line = False
    return lines

def adaptive_middle_band_for_line(ink_mask, y1, y2):
    """Adaptive middle zone detection using peak ink density."""
    line = ink_mask[y1:y2, :]
    vdens = np.sum(line > 0, axis=1).astype(np.float32)
    if vdens.max() == 0:
        return None

    smoothed = cv2.GaussianBlur(vdens.reshape(-1, 1), (5, 1), 0).flatten()
    peak_idx = np.argmax(smoothed)
    peak_val = smoothed[peak_idx]

    # expand around 50% of peak
    top_idx = peak_idx
    while top_idx > 0 and smoothed[top_idx] > 0.5 * peak_val:
        top_idx -= 1

    bottom_idx = peak_idx
    while bottom_idx < len(smoothed) - 1 and smoothed[bottom_idx] > 0.5 * peak_val:
        bottom_idx += 1

    lower_idx = max(0, top_idx)
    upper_idx = min(len(smoothed) - 1, bottom_idx)

    if upper_idx - lower_idx < 5:
        return None
    return (y1 + lower_idx, y1 + upper_idx)

# ---------- main analysis ----------
def analyze_letter_size_middle_zone(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Image not found:", img_path)
        return

    base = cv2.medianBlur(img, 5)
    binary = ensure_binary_white_ink(base)

    coords = cv2.findNonZero(binary)
    if coords is None:
        print("❌ No ink detected.")
        return
    x, y, w, h = cv2.boundingRect(coords)

    px_per_mm = hybrid_px_per_mm(img, binary, (x, y, w, h))
    three_mm_px = 3.0 * px_per_mm

    lines = find_lines(binary)
    if not lines:
        print("❌ No handwriting lines detected.")
        return

    midbands = []
    for (y1, y2) in lines:
        mb = adaptive_middle_band_for_line(binary, y1, y2)
        if mb:
            midbands.append(mb)

    if not midbands:
        print("❌ Couldn’t determine middle zones.")
        return

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mid_heights = []
    mid_boxes = []

    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bh < 6 or bh > 160:
            continue
        cy = by + bh // 2
        if not any(m1 <= cy <= m2 for (m1, m2) in midbands):
            continue
        if bw < 2 and bh < three_mm_px * 0.4:
            continue
        mid_heights.append(bh)
        mid_boxes.append((bx, by, bw, bh))

    if not mid_heights:
        print("❌ No mid-zone letters detected.")
        return

    h_arr = np.array(mid_heights, dtype=np.float32)
    q1, q3 = np.percentile(h_arr, [25, 75])
    iqr_mask = (h_arr >= q1) & (h_arr <= q3)
    xheight_px = np.median(h_arr[iqr_mask]) if iqr_mask.any() else np.median(h_arr)

    fit_count = sum(1 for (_, _, _, bh) in mid_boxes if bh <= three_mm_px)
    total = len(mid_boxes)
    fit_ratio = fit_count / total
    median_mm = xheight_px / px_per_mm
    avg_mm = float(np.mean(h_arr)) / px_per_mm

    # ----- refined classification -----
    # --- Step 6: Classification (Simplified — only two traits) ---
    if median_mm <= 3.0:
        label = "Small Letter (≤ 3 mm)"
        trait = "Introvert"
    else:
        label = "Large Letter (> 3 mm)"
        trait = "Extrovert"

    # ---- Terminal output ----
    print("\n📊 TRUE MIDDLE-ZONE LETTER SIZE (adaptive x-height)")
    print("──────────────────────────────────────────────")
    print(f"🔧 Scale estimate: {px_per_mm:.2f} px/mm  (3 mm ≈ {three_mm_px:.1f} px)")
    print(f"📦 Mid-zone letters analyzed: {total}")
    print(f"✅ ≤ 3 mm: {fit_ratio*100:.1f}%")
    print(f"📏 Median x-height: {median_mm:.2f} mm   (avg: {avg_mm:.2f} mm)")
    print(f"🧠 Classification: {label} → {trait}")

    # ---- Save visualization (optional) ----
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for (m1, m2) in midbands:
        cv2.rectangle(vis, (0, m1), (vis.shape[1]-1, m2), (255, 200, 0), 1)
    for (bx, by, bw, bh) in mid_boxes:
        color = (0, 255, 0) if bh <= three_mm_px else (0, 0, 255)
        cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), color, 1)

    out_path = os.path.join(os.path.dirname(img_path), "adaptive_middlezone_overlay.png")
    cv2.imwrite(out_path, vis)
    print(f"💾 Overlay saved: {out_path}")

# ---------- run ----------
if __name__ == "__main__":
    p = input("📂 Enter path to handwriting image: ").strip()
    analyze_letter_size_middle_zone(p)