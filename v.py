import cv2
import numpy as np
import os

def detect_letter_size(img_path):
    # --- Step 1: Load image ---
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError("❌ Image not found. Check your path.")

    # --- Step 2: Grayscale ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Step 3: Shadow removal ---
    background = cv2.medianBlur(gray, 35)
    diff = 255 - cv2.absdiff(gray, background)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # --- Step 4: Denoise & threshold ---
    blurred = cv2.GaussianBlur(norm, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 11
    )

    # --- Step 5: Morphological cleanup ---
    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.dilate(clean, kernel, iterations=1)

    # --- Step 6: Find contours ---
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    heights = [cv2.boundingRect(c)[3] for c in contours if 8 < cv2.boundingRect(c)[3] < 120]
    if not heights:
        raise ValueError("❌ No valid contours found for letter size measurement.")

    # --- Step 7: Compute average/median height ---
    median_height = np.median(heights)
    avg_height = np.mean(heights)

    # --- Step 8: Convert to mm (approx 5.5 px/mm for phone photos) ---
    px_per_mm = 5.5
    median_mm = median_height / px_per_mm
    avg_mm = avg_height / px_per_mm

    # --- Step 9: Classify ---
    if median_mm <= 3.0:
        classification = "Small Letter (below 3 mm)"
        trait = "Introvert"
    else:
        classification = "Large Letter (above 3 mm)"
        trait = "Extrovert"

    # --- Step 10: Save processed binary image ---
    output_path = os.path.join(os.path.dirname(img_path), "clean_binary.png")
    cv2.imwrite(output_path, clean)

    # --- Step 11: Print results ---
    print("\n📊 HANDWRITING LETTER SIZE ANALYSIS")
    print("────────────────────────────────────")
    print(f"📏 Median Letter Height (px): {median_height:.2f}")
    print(f"📏 Average Letter Height (px): {avg_height:.2f}")
    print(f"✅ Median Letter Height (mm): {median_mm:.2f}")
    print(f"✅ Average Letter Height (mm): {avg_mm:.2f}")
    print(f"🧠 Classification: {classification} → Personality: {trait}")
    print(f"💾 Preprocessed binary image saved at: {output_path}\n")


if __name__ == "__main__":
    img_path = input("📂 Enter path to handwriting image: ").strip()
    if not os.path.exists(img_path):
        print("❌ Image not found:", img_path)
    else:
        detect_letter_size(img_path)