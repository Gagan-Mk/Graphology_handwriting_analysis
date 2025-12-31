import cv2
import numpy as np

# --- Step 1: Load grayscale image ---
img = cv2.imread("/Users/gagan/Desktop/trail-images/rahul.png", 0)
if img is None:
    raise FileNotFoundError("❌ Image not found. Check path or filename.")

# --- Step 2: Contrast enhancement (important for page edge) ---
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
enhanced = clahe.apply(img)

# --- Step 3: Detect A4 page boundary using Canny ---
edges = cv2.Canny(enhanced, 50, 150)
edges = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=2)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not contours:
    raise ValueError("❌ No page contour found. Try adjusting lighting or contrast.")

# --- Step 4: Pick largest contour (the A4 sheet) ---
page_contour = max(contours, key=cv2.contourArea)
x_page, y_page, w_page, h_page = cv2.boundingRect(page_contour)

# --- Step 5: Detect handwriting ---
blur = cv2.GaussianBlur(img, (5, 5), 0)
binary = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    15, 9
)

kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
binary = cv2.dilate(binary, kernel, iterations=1)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_contours = [
    c for c in contours
    if cv2.contourArea(c) > 50
    and cv2.boundingRect(c)[0] > x_page + w_page * 0.01
    and cv2.boundingRect(c)[1] > y_page + h_page * 0.05
]

if not valid_contours:
    raise ValueError("❌ No handwriting contours detected after filtering.")

# --- Step 6: Compute margins relative to the *left page edge* ---
topmost = min([cv2.boundingRect(c)[1] for c in valid_contours]) - y_page
leftmost_text = min([cv2.boundingRect(c)[0] for c in valid_contours])
left_margin_px = leftmost_text - x_page  # <-- distance from left edge of A4

# Convert to inches
inch_px = 118
top_margin_in = topmost / inch_px
left_margin_in = left_margin_px / inch_px

# --- Step 7: Debug visualization ---
debug = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# Page boundary (blue)
cv2.rectangle(debug, (x_page, y_page), (x_page + w_page, y_page + h_page), (255, 0, 0), 2)
# Top margin (red)
cv2.line(debug, (x_page, y_page + int(topmost)), (x_page + w_page, y_page + int(topmost)), (0, 0, 255), 2)
# Left margin (green) → now correctly from left edge of page
cv2.line(debug, (x_page + int(left_margin_px), y_page), (x_page + int(left_margin_px), y_page + h_page), (0, 255, 0), 2)

cv2.imshow("Detected Top & Left Margins", debug)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("📏 Page boundary detected:")
print(f"Top margin (inches): {round(top_margin_in, 2)}")
print(f"Left margin (inches): {round(left_margin_in, 2)}")