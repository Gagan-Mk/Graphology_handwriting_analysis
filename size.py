import cv2
import numpy as np
import os

def analyze_letter_size(img_path):
    # --- Step 1: Load image ---
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("❌ Image not found.")

    # --- Step 2: Preprocess with multiple fallback methods ---
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    h, w = img.shape
    max_contour_area = h * w * 0.95  # Reject contours larger than 95% of image (likely the whole page)
    binary = None
    contours = None
    
    # Method 1: Adaptive threshold (works best for photos with varying lighting)
    try:
        binary = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15, 9
        )
        # Morphological operations to separate letters
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # Use RETR_LIST to get all contours, not just external
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Filter out huge contours (likely the whole page)
        if contours:
            contours = [c for c in contours if cv2.contourArea(c) < max_contour_area]
    except:
        pass
    
    # Method 2: Otsu threshold (adaptive)
    if not contours or len(contours) == 0:
        try:
            _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # Morphological operations
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contours = [c for c in contours if cv2.contourArea(c) < max_contour_area]
        except:
            pass
    
    # Method 3: Try with inverted Otsu
    if not contours or len(contours) == 0:
        try:
            _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Check if we need to invert (white background)
            if np.sum(binary == 255) > np.sum(binary == 0):
                binary = 255 - binary
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contours = [c for c in contours if cv2.contourArea(c) < max_contour_area]
        except:
            pass
    
    # Method 4: Simple threshold with inversion
    if not contours or len(contours) == 0:
        try:
            _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contours = [c for c in contours if cv2.contourArea(c) < max_contour_area]
        except:
            pass

    if not contours or len(contours) == 0:
        raise ValueError("❌ No contours detected in handwriting. Try adjusting image quality or preprocessing.")

    # --- Step 3: Determine scale (auto-calibrate using A4 width ≈ 210 mm) ---
    if binary is not None:
        h, w = binary.shape
    else:
        h, w = img.shape
    px_per_mm = w / 210.0   # dynamic scaling based on photo width
    
    # Debug: Print contour statistics
    all_heights = [cv2.boundingRect(c)[3] for c in contours]
    all_areas = [cv2.contourArea(c) for c in contours]
    if all_heights:
        print(f"[DEBUG] Found {len(contours)} contours")
        print(f"[DEBUG] Height range: {min(all_heights)} - {max(all_heights)} pixels")
        print(f"[DEBUG] Area range: {min(all_areas):.1f} - {max(all_areas):.1f} pixels²")

    # --- Step 4: Measure each letter height with progressive filtering ---
    heights_mm = []
    heights_px = []
    
    # Try with original filter first
    for c in contours:
        x, y, w_box, h_box = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        # Filter by height and area (area helps filter out noise)
        if 8 < h_box < 120 and area > 20:
            h_mm = h_box / px_per_mm
            heights_mm.append(h_mm)
            heights_px.append(h_box)
    
    # If no valid letters, try with relaxed filter
    if not heights_mm:
        for c in contours:
            x, y, w_box, h_box = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if 5 < h_box < 150 and area > 10:  # relaxed filter
                h_mm = h_box / px_per_mm
                heights_mm.append(h_mm)
                heights_px.append(h_box)
    
    # If still no valid letters, try with even more relaxed filter
    if not heights_mm:
        for c in contours:
            x, y, w_box, h_box = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if 3 < h_box < 200 and area > 5:  # very relaxed filter
                h_mm = h_box / px_per_mm
                heights_mm.append(h_mm)
                heights_px.append(h_box)
    
    # Last resort: use aspect ratio to filter letters (letters are usually taller than wide)
    if not heights_mm:
        for c in contours:
            x, y, w_box, h_box = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            aspect_ratio = h_box / max(w_box, 1)
            # Letters typically have aspect ratio > 0.5 (taller than wide or square-ish)
            if h_box > 3 and area > 5 and aspect_ratio > 0.3:
                h_mm = h_box / px_per_mm
                # Only include if it's a reasonable letter size (0.5mm to 15mm)
                if 0.5 < h_mm < 15:
                    heights_mm.append(h_mm)
                    heights_px.append(h_box)

    if not heights_mm:
        print(f"[DEBUG] Image dimensions: {w} x {h} pixels")
        print(f"[DEBUG] Estimated px/mm: {px_per_mm:.2f}")
        print(f"[DEBUG] Total contours found: {len(contours)}")
        if all_heights:
            print(f"[DEBUG] Height distribution: min={min(all_heights)}, max={max(all_heights)}, median={np.median(all_heights):.1f}")
        raise ValueError("❌ No valid letters detected. The image may not contain readable handwriting or needs better preprocessing.")
    
    print(f"[DEBUG] Successfully detected {len(heights_mm)} valid letters")

    # --- Step 6: Compare against 3 mm threshold ---
    large_letters = sum(1 for h in heights_mm if h > 3)
    small_letters = sum(1 for h in heights_mm if h <= 3)
    total_letters = len(heights_mm)
    large_ratio = large_letters / total_letters

    # --- Step 7: Decide classification ---
    if large_ratio > 0.5:
        classification = "Large Letter (above 3 mm)"
        trait = "Extrovert"
    else:
        classification = "Small Letter (below 3 mm)"
        trait = "Introvert"

    avg_height = np.mean(heights_mm)
    median_height = np.median(heights_mm)

    # --- Step 8: Print results ---
    print("\n📊 HANDWRITING LETTER SIZE ANALYSIS")
    print("────────────────────────────────────")
    print(f"📏 Total Letters Analyzed: {total_letters}")
    print(f"📈 % Above 3 mm: {large_ratio * 100:.1f}%")
    print(f"📐 Median Letter Height: {median_height:.2f} mm")
    print(f"📐 Average Letter Height: {avg_height:.2f} mm")
    print(f"🧠 Classification: {classification} → Personality: {trait}")

    # --- Step 9: Save debug image (optional visualization) ---
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    color_large = (0, 0, 255)
    color_small = (0, 255, 0)

    # Draw all valid letters that were used in analysis
    if binary is not None:
        for c in contours:
            x, y, w_box, h_box = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            # Use the same filter criteria as in analysis
            if ((8 < h_box < 120 and area > 20) or 
                (5 < h_box < 150 and area > 10) or 
                (3 < h_box < 200 and area > 5)):
                h_mm = h_box / px_per_mm
                # Only draw if it's in the valid range we actually used
                if 0.5 < h_mm < 20:  # reasonable letter size range
                    color = color_large if h_mm > 3 else color_small
                    cv2.rectangle(vis, (x, y), (x + w_box, y + h_box), color, 1)

    output_path = os.path.join(os.path.dirname(img_path), "letter_size_boxes.png")
    cv2.imwrite(output_path, vis)
    print(f"💾 Debug visualization saved at: {output_path}\n")

# --- Run manually ---
if __name__ == "__main__":
    img_path = input("📂 Enter path to binary handwriting image: ").strip()
    if not os.path.exists(img_path):
        print("❌ Image not found:", img_path)
    else:
        analyze_letter_size(img_path)