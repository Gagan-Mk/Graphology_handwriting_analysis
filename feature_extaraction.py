import cv2
import numpy as np
import pandas as pd
import os
import glob
from scipy.stats import linregress
from baseline import get_baseline_angle as baseline_get_baseline_angle

# ============================================================
# 0️⃣  IMAGE PREPROCESSING → Deskew, Threshold, Normalize
# ============================================================
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 15, 10)

    coords = np.column_stack(np.where(img > 0))
    if coords.size == 0:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img

# ============================================================
# 1️⃣ SLANT ANGLE → Emotional Scale
# ============================================================
def get_slant_angle(img):
    edges = cv2.Canny(img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=40, maxLineGap=10)
    if lines is None:
        return 90.0, "Vertical (AB)", "Balanced"

    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -80 < angle < 80:
            angles.append(angle)
    if not angles:
        return 90.0, "Vertical (AB)", "Balanced"

    avg_angle = np.mean(angles)
    deg = (avg_angle + 180) % 180

    if 56 <= deg < 90:
        return round(deg, 2), "Leftward (FA)", "Suppressed"
    elif 90 <= deg < 112:
        return round(deg, 2), "Vertical (AB)", "Balanced"
    elif 112 <= deg < 125:
        return round(deg, 2), "Slight Right (BC)", "Balanced"
    elif 125 <= deg < 135:
        return round(deg, 2), "Moderate Right (CD)", "Expressive"
    elif 135 <= deg < 150:
        return round(deg, 2), "Extreme Right (DE)", "Expressive"
    elif deg >= 150:
        return round(deg, 2), "Over Expressive (E+)", "Expressive"
    else:
        return round(deg, 2), "Vertical", "Balanced"

# ============================================================
# 2️⃣ MARGINS → Orientation Trait (Left + Top → Orientation)
# ============================================================
# def get_margins(img):
#     import cv2
#     import numpy as np
#     import os

#     # --- Step 1: Ensure grayscale ---
#     if len(img.shape) == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # --- Step 2: Smooth + adaptive threshold ---
#     blur = cv2.GaussianBlur(img, (5, 5), 0)
#     binary = cv2.adaptiveThreshold(
#         blur, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV,
#         15, 9
#     )

#     # --- Step 3: Morphological cleanup ---
#     kernel = np.ones((3, 3), np.uint8)
#     binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
#     binary = cv2.dilate(binary, kernel, iterations=1)

#     # --- Step 4: Find contours ---
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         raise ValueError("❌ No contours found. Check threshold or ink color.")

#     height, width = img.shape

#     # --- Step 5: Filter contours (ignore top/left noise) ---
#     valid_contours = [
#         c for c in contours
#         if cv2.contourArea(c) > 50
#         and cv2.boundingRect(c)[1] > height * 0.05
#         and cv2.boundingRect(c)[0] > width * 0.01
#     ]

#     if not valid_contours:
#         raise ValueError("❌ No valid contours after filtering. Try lowering area threshold.")

#     # --- Step 6: Get topmost and leftmost writing pixels ---
#     topmost = min([cv2.boundingRect(c)[1] for c in valid_contours])
#     leftmost = min([cv2.boundingRect(c)[0] for c in valid_contours])

#     # --- Step 7: Convert to inches (A4 ≈ 118 px per inch) ---
#     inch_px = 118
#     top_margin_in = topmost / inch_px
#     left_margin_in = leftmost / inch_px

#     # --- Step 8: Classify margins ---
#     top_ratio = topmost / height
#     left_ratio = leftmost / width

#     top_type = "Large" if top_ratio > 0.07 else "Small"
#     left_type = "Large" if left_ratio > 0.07 else "Small"

#     top_trait = "Calm" if top_type == "Large" else "Ambitious"
#     left_trait = "Detached" if left_type == "Large" else "Attached"

#     # --- Step 9: Visual debug overlay (optional, saved automatically) ---
#     debug = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     cv2.line(debug, (0, topmost), (width, topmost), (0, 0, 255), 2)   # Red = top
#     cv2.line(debug, (leftmost, 0), (leftmost, height), (0, 255, 0), 2)  # Green = left

#     out_path = os.path.join(os.getcwd(), "debug_margins.png")
#     cv2.imwrite(out_path, debug)
#     print(f"💾 Margin debug image saved at: {out_path}")

#     # --- Step 10: Return values (used in your main feature pipeline) ---
#     return (
#         round(left_margin_in, 2), left_type, left_trait,
#         round(top_margin_in, 2), top_type, top_trait
#     )



def get_margins(img_or_path):
    import cv2, os, tempfile
    from new0margin import get_margins_from_image

    # Handle both path and ndarray inputs
    if isinstance(img_or_path, str):
        img_path = img_or_path
    else:
        tmp_path = os.path.join(tempfile.gettempdir(), "temp_margin_input.png")
        cv2.imwrite(tmp_path, img_or_path)
        img_path = tmp_path

    # Extract margins directly from new0margin.py (already in inches)
    left_margin_in, top_margin_in = get_margins_from_image(img_path)
    print(f"[DEBUG] Margins from new0margin: Left={left_margin_in} Top={top_margin_in}")
    # Classify based on true inch values
    left_type = "Large" if left_margin_in > 1.0 else "Small"
    top_type = "Large" if top_margin_in > 1.0 else "Small"

    left_trait = "Detached" if left_type == "Large" else "Attached"
    top_trait = "Calm" if top_type == "Large" else "Ambitious"

    print(f"✅ Left Margin: {left_margin_in} in ({left_type}) | Top Margin: {top_margin_in} in ({top_type})")

    return (
        round(left_margin_in, 2), left_type, left_trait,
        round(top_margin_in, 2), top_type, top_trait
    )
    
# ============================================================
# 3️⃣ BASELINE ANGLE → Emotional Outlook (delegates to baseline.py)
# ============================================================
def get_baseline_angle(image_path):
    """
    Fetch baseline angle results from baseline.py using the image path.
    Returns: (angle_degrees, feature_label, trait_label)
    """
    try:
        angle, feature, trait = baseline_get_baseline_angle(image_path, debug=False)
        return angle, feature, trait
    except Exception as e:
        print(f"[DEBUG] Baseline fallback due to error: {e}")
        return 0.0, "Straight", "Balanced"






# ============================================================
# 4️⃣ LETTER SIZE → Personality Trait (Integrated from size:py)
# ============================================================

def get_letter_size(img_or_path):
    """
    Analyze letter size using algorithm from size:py.
    Accepts either image path (str) or preprocessed image (ndarray).
    Returns: (median_mm, label, trait)
    """
    # Handle both path and ndarray inputs (like get_margins does)
    if isinstance(img_or_path, str):
        # Load original image from path
        img = cv2.imread(img_or_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 3.0, "Small Letter", "Introvert"
    else:
        # If preprocessed image is passed, we need the original
        # For now, try to work with what we have, but it may not be ideal
        img = img_or_path
        # Check if image is already binary (only 0 and 255 values)
        unique_vals = np.unique(img)
        if len(unique_vals) <= 2 and (0 in unique_vals and 255 in unique_vals):
            # It's binary, we can't do proper thresholding, so return default
            # In this case, we should ideally get the original image path
            return 3.0, "Small Letter", "Introvert"
    
    # --- Step 1: Preprocess with multiple fallback methods ---
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
        # Fallback: return default values
        return 3.0, "Small Letter", "Introvert"

    # --- Step 2: Determine scale (auto-calibrate using A4 width ≈ 210 mm) ---
    px_per_mm = w / 210.0   # dynamic scaling based on photo width

    # --- Step 3: Measure each letter height with progressive filtering ---
    heights_mm = []
    
    # Try with original filter first
    for c in contours:
        x, y, w_box, h_box = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        # Filter by height and area (area helps filter out noise)
        if 8 < h_box < 120 and area > 20:
            h_mm = h_box / px_per_mm
            heights_mm.append(h_mm)
    
    # If no valid letters, try with relaxed filter
    if not heights_mm:
        for c in contours:
            x, y, w_box, h_box = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if 5 < h_box < 150 and area > 10:  # relaxed filter
                h_mm = h_box / px_per_mm
                heights_mm.append(h_mm)
    
    # If still no valid letters, try with even more relaxed filter
    if not heights_mm:
        for c in contours:
            x, y, w_box, h_box = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if 3 < h_box < 200 and area > 5:  # very relaxed filter
                h_mm = h_box / px_per_mm
                heights_mm.append(h_mm)
    
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

    if not heights_mm:
        # Fallback: return default values
        return 3.0, "Small Letter", "Introvert"

    # --- Step 4: Calculate statistics and classify ---
    median_mm = np.median(heights_mm)
    avg_mm = np.mean(heights_mm)
    
    # Use median for classification (more robust than mean)
    if median_mm <= 3.0:
        return round(median_mm, 2), "Small Letter", "Introvert"
    else:
        return round(median_mm, 2), "Large Letter", "Extrovert"

# ============================================================
# 5️⃣ WORD SPACING → Social Interaction (Updated from Wordsp.py - 2 Categories)
# ============================================================
def get_word_spacing(img, img_path=None):
    """
    Extract word spacing using adaptive kernel and 2-category classification.
    Uses algorithm from Wordsp.py: Sociable (Wide) or Reserved (Narrow).
    
    Args:
        img: Preprocessed binary image
        img_path: Optional image path for visualization (not used in main extraction)
    
    Returns:
        tuple: (ratio, feature_label, trait_label)
        - ratio: Word spacing ratio
        - feature_label: "Wide" or "Narrow"
        - trait_label: "Sociable" or "Reserved"
    """
    # Adaptive kernel size based on image dimensions
    k_w = max(25, img.shape[1] // 100)
    k_h = max(5, img.shape[0] // 300)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Find word contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 1.0, "Narrow", "Reserved"

    boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: (b[1], b[0]))

    # Measure gaps between words on same line
    gaps, prev_y, prev_x, prev_w = [], None, None, None
    for (x, y, w, h) in boxes:
        if prev_y is None:
            prev_y, prev_x, prev_w = y, x, w
            continue
        # Same line if vertical difference is less than 1.5x the height
        if abs(y - prev_y) < h * 1.5:
            gap = x - (prev_x + prev_w)
            if 5 < gap < 150:  # Reasonable gap size
                gaps.append(gap)
        prev_y, prev_x, prev_w = y, x, w

    if not gaps:
        return 1.0, "Narrow", "Reserved"

    # Compute statistics using percentile-based estimation
    avg_gap = np.median(gaps)
    word_widths = [w for (_, _, w, _) in boxes]
    W_est = np.percentile(word_widths, 90) * 1.2  # Use 90th percentile * 1.2
    r = avg_gap / W_est
    r = float(np.clip(r, 0.1, 2.0))

    # Classification: Only 2 categories (Sociable/Reserved)
    if r > 1.1:
        return round(r, 2), "Wide", "Sociable"
    else:
        return round(r, 2), "Narrow", "Reserved"

# ============================================================
# 6️⃣ LINE SPACING → Emotional Space
# ============================================================
def get_line_spacing(img):
    proj = np.sum(img, axis=1)
    y_idx = np.where(proj > np.mean(proj) * 0.3)[0]
    if len(y_idx) < 10:
        return 1.0, "Moderate", "Balanced"

    lines, start = [], y_idx[0]
    for i in range(1, len(y_idx)):
        if y_idx[i] - y_idx[i - 1] > 5:
            lines.append((start, y_idx[i - 1]))
            start = y_idx[i]
    lines.append((start, y_idx[-1]))

    if len(lines) < 2:
        return 1.0, "Moderate", "Balanced"

    distances = [lines[i + 1][0] - lines[i][1] for i in range(len(lines) - 1)]
    avg_space = np.median(distances)
    xheight = np.median([y2 - y1 for (y1, y2) in lines])
    ratio = max(avg_space / xheight, 0.5)

    if ratio > 1.2:
        return round(ratio, 2), "Wide", "Distant"
    elif ratio < 0.8:
        return round(ratio, 2), "Narrow", "Clingy"
    else:
        return round(ratio, 2), "Moderate", "Balanced"

# ============================================================
# 7️⃣ FINAL FEATURE EXTRACTION + TRAIT MAPPING
# ============================================================
def extract_all_features(img_path):
    img = preprocess_image(img_path)
    data = {"ImageName": os.path.basename(img_path)}

    data["SlantAngle"], data["SlantFeature"], data["EmotionalScale"] = get_slant_angle(img)
    # Pass img_path to get_margins for proper A4 detection using new0margin.py
    (data["LeftMarginIn"], data["LeftMarginType"], left_trait,
     data["TopMarginIn"], data["TopMarginType"], top_trait) = get_margins(img_path)

    # ✅ Orientation derived from both left & top traits
    if left_trait == "Attached" and top_trait == "Ambitious":
        orientation_label = "Goal-Oriented"
    elif left_trait == "Attached" and top_trait == "Calm":
        orientation_label = "Stable"
    elif left_trait == "Detached" and top_trait == "Ambitious":
        orientation_label = "Independent Thinker"
    elif left_trait == "Detached" and top_trait == "Calm":
        orientation_label = "Reserved"
    else:
        orientation_label = "Balanced"

    data["Orientation"] = orientation_label

    data["BaselineAngle"], data["BaselineFeature"], data["EmotionalOutlook"] = get_baseline_angle(img_path)
    # Pass img_path to get_letter_size for proper preprocessing (like size:py does)
    data["LetterSizeMM"], data["LetterFeature"], data["LetterSizeTrait"] = get_letter_size(img_path)
    # Pass img_path to get_word_spacing (uses Wordsp.py algorithm with 2 categories)
    data["WordSpacingRatio"], data["WordFeature"], word_trait = get_word_spacing(img, img_path)
    data["LineSpacingRatio"], data["LineFeature"], line_trait = get_line_spacing(img)

    # Updated logic for 2-category word spacing (Sociable/Reserved only)
    if word_trait == "Reserved" and line_trait == "Balanced":
        data["SocialIsolation"] = "Reserved"
    elif word_trait == "Sociable" and line_trait == "Balanced":
        data["SocialIsolation"] = "Sociable"
    elif word_trait == "Reserved" and line_trait == "Clingy":
        data["SocialIsolation"] = "Reserved"
    elif word_trait == "Sociable" and line_trait == "Clingy":
        data["SocialIsolation"] = "Clingy"
    elif word_trait == "Reserved" and line_trait == "Distant":
        data["SocialIsolation"] = "Reserved"
    elif word_trait == "Sociable" and line_trait == "Distant":
        data["SocialIsolation"] = "Balanced"
    else:
        # Default to word_trait (Sociable or Reserved)
        data["SocialIsolation"] = word_trait

    even_baseline = abs(data["BaselineAngle"]) < 2
    letter_size = data["LetterSizeMM"]
    ratio = data["LineSpacingRatio"]
    if letter_size <= 3 and even_baseline:
        data["Concentration"] = "Focused"
    elif letter_size <= 3 and not even_baseline:
        data["Concentration"] = "Tense"
    elif letter_size > 3 and even_baseline:
        data["Concentration"] = "Relaxed"
    else:
        if ratio > 1.2:
            data["Concentration"] = "Relaxed"
        elif ratio < 0.8:
            data["Concentration"] = "Focused"
        else:
            data["Concentration"] = "Distracted"

    return pd.DataFrame([data])

# ============================================================
# 8️⃣ EXECUTION ON ALL IMAGES IN FOLDER
# ============================================================
if __name__ == "__main__":
    folder_path = r"/Users/gagan/Desktop/images"
    output_csv = os.path.join(folder_path, "features_mapped.csv")

    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.heif"):
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    if not image_paths:
        print("⚠️ No images found in:", folder_path)
        raise SystemExit

    all_rows = []
    for img_path in image_paths:
        print(f"\n🖼️ Processing: {os.path.basename(img_path)}")
        try:
            row_df = extract_all_features(img_path)
            all_rows.append(row_df)
            cols_to_show = [
                "ImageName",
                "SlantAngle", "SlantFeature", "EmotionalScale",
                "LeftMarginIn", "LeftMarginType", "TopMarginIn", "TopMarginType", "Orientation",
                "BaselineAngle", "BaselineFeature", "EmotionalOutlook",
                "LetterSizeMM", "LetterFeature", "LetterSizeTrait",
                "WordSpacingRatio", "WordFeature", "SocialIsolation",
                "LineSpacingRatio", "LineFeature", "Concentration"
            ]
            print(row_df[[c for c in cols_to_show if c in row_df.columns]].T)
        except Exception as e:
            print(f"❌ Error {os.path.basename(img_path)}: {e}")

    if not all_rows:
        print("⚠️ No valid images processed.")
        raise SystemExit

    final_df = pd.concat(all_rows, ignore_index=True)
    ordered_cols = [
        "ImageName",
        "SlantAngle", "SlantFeature", "EmotionalScale",
        "LeftMarginIn", "LeftMarginType", "TopMarginIn", "TopMarginType", "Orientation",
        "BaselineAngle", "BaselineFeature", "EmotionalOutlook",
        "LetterSizeMM", "LetterFeature", "LetterSizeTrait",
        "WordSpacingRatio", "WordFeature", "SocialIsolation",
        "LineSpacingRatio", "LineFeature", "Concentration"
    ]
    final_df = final_df[ordered_cols]
    final_df.to_csv(output_csv, index=False)
    print("\n✅ Saved features and traits to:", output_csv)
