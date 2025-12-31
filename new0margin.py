import cv2
import numpy as np
import os

def get_margins_from_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"❌ Image not found: {img_path}")

    # --- Step 1: Detect and crop A4 sheet from background ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    sheet_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            sheet_contour = approx
            break

    if sheet_contour is not None:
        pts = sheet_contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxWidth = int(max(widthA, widthB))
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        cropped = cv2.warpPerspective(gray, M, (maxWidth, maxHeight))
        print("🧾 A4 sheet detected and cropped successfully.")
    else:
        cropped = gray.copy()
        print("⚠️ No A4 contour detected — using full image.")

    # --- Step 2: Preprocess cropped sheet with multiple fallback methods ---
    blur = cv2.GaussianBlur(cropped, (5, 5), 0)
    # Improve local contrast for faint handwriting
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        blur = clahe.apply(blur)
    except:
        pass
    
    # Try multiple thresholding methods
    binary = None
    contours = None
    
    # Method 1: Adaptive threshold (default)
    try:
        binary = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15, 9
        )
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        # avoid excessive dilation that can erase thin strokes
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    except:
        pass
    
    # Method 2: Otsu threshold (fallback if adaptive fails)
    if not contours or len(contours) == 0:
        try:
            _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        except:
            pass
    
    # Method 3: Inverted adaptive threshold (if image has dark background)
    if not contours or len(contours) == 0:
        try:
            binary = cv2.adaptiveThreshold(
                blur, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                15, 9
            )
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        except:
            pass
    
    # Method 4: Simple threshold with lower threshold value
    if not contours or len(contours) == 0:
        try:
            _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        except:
            pass
    
    # Method 5: Canny edge detection as last resort
    if not contours or len(contours) == 0:
        try:
            edges = cv2.Canny(blur, 50, 150)
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        except:
            pass

    # --- Step 3: Detect contours for writing ---
    if not contours or len(contours) == 0:
        raise ValueError("❌ No writing detected on page. Try adjusting image quality or lighting.")

    height, width = cropped.shape
    
    # Try with original area threshold first
    valid_contours = [
        c for c in contours
        if cv2.contourArea(c) > 50
        and cv2.boundingRect(c)[1] > height * 0.05
        and cv2.boundingRect(c)[0] > width * 0.01
    ]
    
    # If no valid contours, try with lower area threshold
    if not valid_contours:
        valid_contours = [
            c for c in contours
            if cv2.contourArea(c) > 20
            and cv2.boundingRect(c)[1] > height * 0.02
            and cv2.boundingRect(c)[0] > width * 0.005
        ]
    
    # If still no valid contours, try with even lower threshold
    if not valid_contours:
        valid_contours = [
            c for c in contours
            if cv2.contourArea(c) > 10
        ]

    if not valid_contours:
        raise ValueError("❌ No valid contours after filtering. The image may not contain readable handwriting.")

    # --- Step 4: Find top and left writing edges ---
    topmost = min([cv2.boundingRect(c)[1] for c in valid_contours])
    leftmost = min([cv2.boundingRect(c)[0] for c in valid_contours])

    # --- Step 5: Convert pixels → inches dynamically ---
    inch_px = height / 11.7
    top_margin_in = topmost / inch_px
    left_margin_in = leftmost / inch_px

    print(f"Topmost pixel: {topmost}")
    print(f"Leftmost pixel: {leftmost}")
    print(f"Image size: {width} x {height}")
    print(f"Top margin (inches): {top_margin_in:.2f}")
    print(f"Left margin (inches): {left_margin_in:.2f}")

    # --- Step 6: Visualization of margins on cropped page ---
    vis = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
    # Draw top margin line (red)
    cv2.line(vis, (0, int(topmost)), (width - 1, int(topmost)), (0, 0, 255), 2)
    cv2.putText(vis, f"Top {top_margin_in:.2f} in", (10, max(int(topmost) - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # Draw left margin line (green)
    cv2.line(vis, (int(leftmost), 0), (int(leftmost), height - 1), (0, 255, 0), 2)
    cv2.putText(vis, f"Left {left_margin_in:.2f} in", (max(int(leftmost) + 10, 10), 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    # Optional: page border for reference
    cv2.rectangle(vis, (0, 0), (width - 1, height - 1), (255, 200, 0), 1)

    # Save and show
    out_path = os.path.join(os.path.dirname(img_path), "margins_debug.png")
    try:
        cv2.imwrite(out_path, vis)
        print(f"💾 Margin visualization saved: {out_path}")
    except Exception as _:
        pass
    try:
        cv2.imshow("Detected Margins", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as _:
        pass

    return round(left_margin_in, 2), round(top_margin_in, 2)


if __name__ == "__main__":
    path = input("Enter image path: ").strip()
    l, t = get_margins_from_image(path)
    print(f"✅ Left margin: {l} in | Top margin: {t} in")