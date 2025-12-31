import cv2
import numpy as np
import os
import sys

# ============================================================
# IMAGE PREPROCESSING → Deskew, Threshold, Normalize
# ============================================================
def preprocess_image(img_path):
    """Reads, binarizes, and deskews the input handwriting image."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"❌ Could not load image: {img_path}")
    
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 10
    )

    # --- Deskew (skip over-rotation) ---
    coords = np.column_stack(np.where(img > 0))
    if coords.size == 0:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if abs(angle) < 1:  # very small tilt, ignore
        return img
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h),
                         flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    return img


# ============================================================
# WORD SPACING EXTRACTION + VISUALIZATION
# ============================================================
def get_word_spacing(img, img_path=None):
    """Extracts word spacing ratio, classifies as Sociable or Reserved, and saves overlay visualization."""
    print("\n🔍 Step 1: Morphological closing to connect letters into words...")
    k_w = max(25, img.shape[1] // 100)
    k_h = max(5, img.shape[0] // 300)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    print(f"   ✅ Adaptive Kernel size: {k_w}×{k_h}")

    # --- Find word contours ---
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"   ✅ Found {len(contours)} word contours")

    if len(contours) == 0:
        return 1.0, "Balanced", "Reserved"

    boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: (b[1], b[0]))

    # --- Measure gaps ---
    gaps, prev_y, prev_x, prev_w = [], None, None, None
    visual_pairs = []
    
    for (x, y, w, h) in boxes:
        if prev_y is None:
            prev_y, prev_x, prev_w = y, x, w
            continue
        if abs(y - prev_y) < h * 1.5:
            gap = x - (prev_x + prev_w)
            if 5 < gap < 150:
                gaps.append(gap)
                visual_pairs.append(((prev_x + prev_w, y + h // 2), (x, y + h // 2)))
        prev_y, prev_x, prev_w = y, x, w

    if not gaps:
        return 1.0, "Balanced", "Reserved"

    # --- Compute stats ---
    avg_gap = np.median(gaps)
    word_widths = [w for (_, _, w, _) in boxes]
    W_est = np.percentile(word_widths, 90) * 1.2
    r = avg_gap / W_est
    r = float(np.clip(r, 0.1, 2.0))

    print(f"   ✅ Spacing Ratio: {r:.2f}")

    # --- Classification (Only 2 Traits) ---
    if r > 1.1:
        feature_label = "Wide"
        trait_label = "Sociable"
    else:
        feature_label = "Narrow"
        trait_label = "Reserved"

    print(f"   ✅ Classified as: {feature_label} ({trait_label})")

    # =====================================================
    # 🖼️ VISUALIZATION
    # =====================================================
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw word boxes
    for (x, y, w, h) in boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Draw lines showing word gaps
    for (p1, p2) in visual_pairs:
        color = (0, 255, 255) if r > 1.1 else (255, 0, 0)  # Yellow for Sociable, Blue for Reserved
        cv2.line(overlay, p1, p2, color, 1)

    # Add label
    text = f"Spacing Ratio: {r:.2f} | {trait_label}"
    color = (0, 255, 255) if trait_label == "Sociable" else (255, 0, 0)
    cv2.putText(overlay, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    # Save visualization
    if img_path:
        out_path = os.path.join(os.path.dirname(img_path), "word_spacing_overlay.png")
        cv2.imwrite(out_path, overlay)
        print(f"\n💾 Visualization saved at: {out_path}")

    return round(r, 2), feature_label, trait_label


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("="*70)
    print("WORD SPACING ANALYSIS — SOCIABLE / RESERVED")
    print("="*70)

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = input("\n📸 Enter the path to the handwriting image: ").strip()

    if not os.path.isabs(img_path):
        img_path = os.path.abspath(img_path)

    if not os.path.exists(img_path):
        print(f"\n❌ Error: Image not found at: {img_path}")
        sys.exit(1)

    print(f"\n🖼️ Processing image: {os.path.basename(img_path)}")

    try:
        img = preprocess_image(img_path)
        ratio, feature_label, trait_label = get_word_spacing(img, img_path)

        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"\n📊 Word Spacing Ratio: {ratio}")
        print(f"🏷️  Feature Classification: {feature_label}")
        print(f"🧠 Personality Trait: {trait_label}")
        print("\n" + "-"*70)
        if trait_label == "Sociable":
            print("   → Wide spacing indicates openness and comfort in social interactions.")
        else:
            print("   → Narrow spacing indicates privacy and emotional reserve.")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Error processing image: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)