def get_margins_from_image(img_path):
    import cv2
    import numpy as np

    img = cv2.imread(img_path, 0)
    if img is None:
        raise FileNotFoundError(f"❌ Image not found: {img_path}")

    # --- Step 1: Smooth + adaptive threshold ---
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 9
    )

    # --- Step 2: Morph cleanup ---
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # --- Step 3: Find contours ---
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("❌ No contours found in image.")

    height, width = img.shape

    valid_contours = [
        c for c in contours
        if cv2.contourArea(c) > 50
        and cv2.boundingRect(c)[1] > height * 0.05
        and cv2.boundingRect(c)[0] > width * 0.01
    ]

    if not valid_contours:
        raise ValueError("❌ No valid contours after filtering.")

    topmost = min([cv2.boundingRect(c)[1] for c in valid_contours])
    leftmost = min([cv2.boundingRect(c)[0] for c in valid_contours])


# --- Step 4: Convert to inches (fixed scale for consistency) ---
    # A4 ≈ 118 px per inch (stable across phone captures)
    inch_px = 118.0
    top_margin_in = topmost / inch_px
    left_margin_in = leftmost / inch_px

    return round(left_margin_in, 2), round(top_margin_in, 2)