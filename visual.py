import cv2
import numpy as np
import math
import os

# ============================================================
# 1️⃣ PREPROCESSING FUNCTION
# ============================================================
def preprocess_image(img_path):
    """Enhance handwriting visibility and binarize image."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"❌ Cannot open {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Step 1: Contrast normalization ---
    gray = cv2.equalizeHist(gray)

    # --- Step 2: Remove noise + smooth ---
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- Step 3: Adaptive threshold ---
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25, 12
    )

    # --- Step 4: Morphological cleanup ---
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return img, gray, binary


# ============================================================
# 2️⃣ BASELINE ANGLE DETECTION WITH VISUALIZATION
# ============================================================
def get_baseline_angle(image_path, debug=True):
    # --- Preprocess the image ---
    img, gray, binary = preprocess_image(image_path)
    h, w = gray.shape

    # --- Step 1: Connect words into lines ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.15), 10))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # --- Step 2: Find contours ---
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("⚠️ No contours found.")
        return 0.0, "Straight", "Balanced"

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    all_angles = []

    # --- Step 3: Try Hough transform ---
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=100, maxLineGap=15)

    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            dx, dy = x2 - x1, y2 - y1
            if abs(dx) < 5:  # ignore near-verticals
                continue
            angle = math.degrees(math.atan2(dy, dx))
            if -25 < angle < 25:
                all_angles.append(angle)
                cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # --- Step 4: Fallback to minAreaRect ---
    if not all_angles:
        print("⚠️ No valid Hough lines — using contour-based angle.")
        for c in contours:
            if cv2.contourArea(c) < 1000:
                continue
            rect = cv2.minAreaRect(c)
            angle = rect[-1]
            if angle < -45:
                angle = 90 + angle
            all_angles.append(angle)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(vis, [box], 0, (255, 0, 0), 2)

    if not all_angles:
        print("⚠️ Still no valid baselines detected.")
        return 0.0, "Straight", "Balanced"

    avg_angle = np.mean(all_angles)

    # --- Step 5: Classification ---
    if avg_angle > 4.0:
        feature, trait = "Descending", "Discouraged"
    elif avg_angle < -4.0:
        feature, trait = "Ascending", "Optimistic"
    else:
        feature, trait = "Straight", "Balanced"

    # --- Step 6: Visualization ---
    if debug:
        text = f"Angle: {avg_angle:.2f}° → {feature} ({trait})"
        cv2.putText(vis, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw a representative red baseline arrow
        center_y = int(h * 0.5)
        length = int(w * 0.4)
        slope = math.tan(math.radians(avg_angle))
        x1, y1 = int(w * 0.3), center_y
        x2, y2 = x1 + length, int(center_y + slope * length)
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 0, 255), 3, tipLength=0.05)

        # Add legend
        cv2.rectangle(vis, (20, 60), (430, 160), (255, 255, 255), -1)
        cv2.putText(vis, "Legend:", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(vis, "Green: Hough lines", (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, "Blue: Contour boxes", (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(vis, "Red: Avg baseline direction", (30, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out_path = os.path.join(os.path.dirname(image_path), "baseline_visual_final.png")
        cv2.imwrite(out_path, vis)
        print(f"💾 Visualization saved: {out_path}")

        cv2.imshow("Baseline Detection Visualization", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return round(avg_angle, 2), feature, trait


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    img_path = input("Enter image path: ").strip()
    angle, feature, trait = get_baseline_angle(img_path, debug=True)
    print(f"\nBaseline Angle: {angle}°")
    print(f"Classification: {feature}")
    print(f"Trait: {trait}")