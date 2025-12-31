import cv2
import numpy as np

# --- Step 1: Load grayscale image ---
img = cv2.imread("/Users/gagan/Desktop/trail-images/Monis-f.jpg", 0)
if img is None:
    raise FileNotFoundError("❌ Image not found. Check path or filename.")

height, width = img.shape

# --- Step 2: Detect page boundaries (A4 sheet) for dynamic scaling ---
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(img)
edges = cv2.Canny(enhanced, 50, 150)
edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

page_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not page_contours:
    raise ValueError("❌ No page contour found. Try improving lighting or crop.")

page_contour = max(page_contours, key=cv2.contourArea)
x_page, y_page, w_page, h_page = cv2.boundingRect(page_contour)

# --- Step 3: Smooth + adaptive threshold for handwriting ---
blur = cv2.GaussianBlur(img, (5, 5), 0)
binary = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    15, 9
)

# --- Step 4: Morph cleanup ---
kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
binary = cv2.dilate(binary, kernel, iterations=1)

# --- Step 5: Find contours of handwriting ---
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    raise ValueError("❌ No handwriting contours detected.")

# --- Step 6: Filter contours inside the detected page ---
valid_contours = []
for c in contours:
    if cv2.contourArea(c) <= 50:
        continue
    x, y, w, h = cv2.boundingRect(c)

    # Require some overlap with page bounds
    if x + w < x_page or x > x_page + w_page:
        continue
    if y + h < y_page or y > y_page + h_page:
        continue

    # Ignore tiny margins around the page to avoid noise
    if y < y_page + h_page * 0.02:
        continue
    if x < x_page + w_page * 0.01:
        continue

    valid_contours.append(c)

if not valid_contours:
    raise ValueError("❌ No valid contours after page-based filtering.")

# --- Step 7: Compute margins relative to the detected page ---
topmost = min([cv2.boundingRect(c)[1] for c in valid_contours]) - y_page
topmost = max(topmost, 0)

leftmost = min([cv2.boundingRect(c)[0] for c in valid_contours]) - x_page
leftmost = max(leftmost, 0)

rightmost_edge = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in valid_contours])
rightmost = (x_page + w_page) - rightmost_edge
rightmost = max(rightmost, 0)

# --- Step 8: Dynamic scale → pixels per inch based on A4 width ---
if w_page > 0:
    px_per_in = w_page / 8.27  # A4 width in inches
else:
    px_per_in = 118.0  # fallback

top_margin_in = topmost / px_per_in
left_margin_in = leftmost / px_per_in
right_margin_in = rightmost / px_per_in

print(f"Top margin (inches): {round(top_margin_in, 2)}")
print(f"Left margin (inches): {round(left_margin_in, 2)}")
print(f"Right margin (inches): {round(right_margin_in, 2)}")

# --- Step 9: Visualization ---
debug = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Draw page boundary
cv2.rectangle(debug, (x_page, y_page), (x_page + w_page, y_page + h_page), (255, 200, 0), 2)

# Left margin line relative to page
left_line_x = int(x_page + leftmost)
cv2.line(debug, (left_line_x, y_page), (left_line_x, y_page + h_page), (0, 255, 0), 2)
cv2.putText(debug, f"Left {left_margin_in:.2f} in", (max(left_line_x + 5, x_page + 5), y_page + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

# Right margin line relative to page
right_line_x = int(x_page + w_page - rightmost)
cv2.line(debug, (right_line_x, y_page), (right_line_x, y_page + h_page), (255, 0, 255), 2)
cv2.putText(debug, f"Right {right_margin_in:.2f} in", (max(right_line_x - 200, x_page + 5), y_page + 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

# Optional: top margin line
top_line_y = int(y_page + topmost)
cv2.line(debug, (x_page, top_line_y), (x_page + w_page, top_line_y), (0, 0, 255), 2)
cv2.putText(debug, f"Top {top_margin_in:.2f} in", (x_page + 5, max(top_line_y - 10, y_page + 25)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow("Page Margins", debug)
cv2.waitKey(0)
cv2.destroyAllWindows()