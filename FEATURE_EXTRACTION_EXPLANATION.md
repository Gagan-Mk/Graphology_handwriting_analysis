# Feature Extraction Implementation Explanation

## Overview

The `feature_extaraction.py` file implements a comprehensive handwriting analysis system that extracts 14 features from handwriting images and maps them to 6 personality traits. It uses computer vision techniques to measure physical characteristics of handwriting.

---

## Technologies Used

### Core Libraries

1. **OpenCV (cv2)** - Computer vision library
   - Image processing, edge detection, contour detection
   - Morphological operations, thresholding
   - Geometric transformations (rotation, warping)

2. **NumPy** - Numerical computing
   - Array operations, mathematical calculations
   - Statistical functions

3. **SciPy** - Scientific computing
   - `linregress` for linear regression (baseline angle calculation)

4. **pandas** - Data manipulation
   - DataFrame creation and management

---

## Feature Extraction Pipeline

```
Input Image
    ↓
[Preprocessing] → Deskew, Threshold, Normalize
    ↓
┌─────────────────────────────────────────┐
│  Parallel Feature Extraction            │
├─────────────────────────────────────────┤
│  1. Slant Angle      → EmotionalScale   │
│  2. Margins          → Orientation      │
│  3. Baseline Angle   → EmotionalOutlook │
│  4. Letter Size      → LetterSizeTrait  │
│  5. Word Spacing     → SocialIsolation  │
│  6. Line Spacing    → Concentration     │
└─────────────────────────────────────────┘
    ↓
[Combine Features] → Final DataFrame
```

---

## 1. Image Preprocessing (`preprocess_image`)

**Purpose**: Prepare image for feature extraction by deskewing and binarizing.

### Steps:

```python
def preprocess_image(img_path):
    # 1. Load as grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # 3. Adaptive thresholding (binarization)
    img = cv2.adaptiveThreshold(
        img, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Adaptive method
        cv2.THRESH_BINARY_INV,            # Invert (white text on black)
        15, 10                            # Block size, C constant
    )
    
    # 4. Deskewing (rotation correction)
    coords = np.column_stack(np.where(img > 0))  # Find text pixels
    angle = cv2.minAreaRect(coords)[-1]           # Get rotation angle
    # Correct angle calculation
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # 5. Rotate image to correct skew
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
    
    return img
```

**Technologies**:
- `cv2.GaussianBlur`: Noise reduction
- `cv2.adaptiveThreshold`: Handles varying lighting conditions
- `cv2.minAreaRect`: Finds minimum bounding rectangle (for angle)
- `cv2.warpAffine`: Geometric transformation (rotation)

**Output**: Binary image (white text on black background), deskewed

---

## 2. Slant Angle Extraction (`get_slant_angle`)

**Purpose**: Measure the angle at which letters slant (left/right) to determine emotional expression.

### Algorithm:

```python
def get_slant_angle(img):
    # 1. Edge detection using Canny
    edges = cv2.Canny(img, 50, 150)  # Low=50, High=150 thresholds
    
    # 2. Line detection using Hough Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,                    # Distance resolution (pixels)
        theta=np.pi/180,          # Angular resolution (1 degree)
        threshold=60,             # Minimum votes for line
        minLineLength=40,         # Minimum line length
        maxLineGap=10             # Maximum gap in line
    )
    
    # 3. Extract angles from detected lines
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -80 < angle < 80:  # Filter horizontal-ish lines
            angles.append(angle)
    
    # 4. Calculate average angle
    avg_angle = np.mean(angles)
    deg = (avg_angle + 180) % 180  # Normalize to 0-180°
    
    # 5. Classify into categories
    if 56 <= deg < 90:
        return deg, "Leftward (FA)", "Suppressed"
    elif 90 <= deg < 112:
        return deg, "Vertical (AB)", "Balanced"
    elif 112 <= deg < 125:
        return deg, "Slight Right (BC)", "Balanced"
    elif 125 <= deg < 135:
        return deg, "Moderate Right (CD)", "Expressive"
    elif 135 <= deg < 150:
        return deg, "Extreme Right (DE)", "Expressive"
    elif deg >= 150:
        return deg, "Over Expressive (E+)", "Expressive"
```

**Technologies**:
- `cv2.Canny`: Edge detection algorithm
- `cv2.HoughLinesP`: Probabilistic Hough Line Transform
- `np.arctan2`: Calculate angle from coordinates
- `np.degrees`: Convert radians to degrees

**Output**: 
- `SlantAngle`: Numeric angle (0-180°)
- `SlantFeature`: Categorical label
- `EmotionalScale`: Personality trait (Suppressed/Balanced/Expressive/Unstable)

**Graphology Rule**: Rightward slant = expressive, Leftward = suppressed, Vertical = balanced

---

## 3. Margin Extraction (`get_margins`)

**Purpose**: Measure left and top margins to determine orientation and attachment style.

### Algorithm:

```python
def get_margins(img_or_path):
    # Uses external module: new0margin.py
    from new0margin import get_margins_from_image
    
    # Extract margins (in inches) using A4 page detection
    left_margin_in, top_margin_in = get_margins_from_image(img_path)
    
    # Classify margin sizes
    left_type = "Large" if left_margin_in > 1.0 else "Small"
    top_type = "Large" if top_margin_in > 1.0 else "Small"
    
    # Map to personality traits
    left_trait = "Detached" if left_type == "Large" else "Attached"
    top_trait = "Calm" if top_type == "Large" else "Ambitious"
    
    return (left_margin_in, left_type, left_trait,
            top_margin_in, top_type, top_trait)
```

**Technologies**:
- A4 page detection (from `new0margin.py`)
- Pixel-to-inch conversion (A4 = 210mm × 297mm)
- Contour detection for page boundaries

**Output**:
- `LeftMarginIn`: Numeric (inches)
- `LeftMarginType`: Small/Large
- `TopMarginIn`: Numeric (inches)
- `TopMarginType`: Small/Large

**Graphology Rule**: 
- Large left margin = detached, independent
- Small left margin = attached, connected
- Large top margin = calm, reserved
- Small top margin = ambitious, goal-oriented

**Orientation** is derived from combination:
- Attached + Ambitious → Goal-Oriented
- Attached + Calm → Stable
- Detached + Ambitious → Independent Thinker
- Detached + Calm → Reserved

---

## 4. Baseline Angle Extraction (`get_baseline_angle`)

**Purpose**: Measure if text lines ascend, descend, or stay straight to determine emotional outlook.

### Algorithm:

```python
def get_baseline_angle(img):
    # 1. Vertical projection (sum pixels in each row)
    proj = np.sum(img, axis=1)  # Sum along horizontal axis
    
    # 2. Find text lines (rows with significant text)
    y_idx = np.where(proj > np.mean(proj) * 0.3)[0]
    
    # 3. Group consecutive rows into lines
    lines = []
    start = y_idx[0]
    for i in range(1, len(y_idx)):
        if y_idx[i] - y_idx[i - 1] > 5:  # Gap > 5 pixels = new line
            lines.append((start, y_idx[i - 1]))
            start = y_idx[i]
    lines.append((start, y_idx[-1]))
    
    # 4. For each line, find baseline (bottom of characters)
    angles = []
    for (y1, y2) in lines:
        line_crop = img[y1:y2, :]  # Extract line region
        contours, _ = cv2.findContours(line_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get bottom points of each character
        pts = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < 20 or h < 8:  # Filter small noise
                continue
            pts.append((x, y + h))  # Bottom-right corner
        
        # 5. Linear regression to find baseline slope
        if len(pts) >= 3:
            x_vals, y_vals = zip(*pts)
            slope, _, _, _, _ = linregress(x_vals, y_vals)
            angles.append(np.degrees(np.arctan(slope)))
    
    # 6. Average angles and classify
    angles = np.clip(angles, -15, 15)  # Limit to ±15°
    avg = np.mean(angles)
    
    if avg > 1.5:
        return avg, "Ascending", "Optimistic"
    elif avg < -1.5:
        return avg, "Descending", "Discouraged"
    else:
        return avg, "Straight", "Balanced"
```

**Technologies**:
- `np.sum(axis=1)`: Vertical projection (histogram)
- `cv2.findContours`: Find character boundaries
- `cv2.boundingRect`: Get character bounding boxes
- `scipy.stats.linregress`: Linear regression to fit baseline
- `np.arctan`: Calculate angle from slope

**Output**:
- `BaselineAngle`: Numeric angle (degrees)
- `BaselineFeature`: Ascending/Descending/Straight
- `EmotionalOutlook`: Optimistic/Discouraged/Balanced

**Graphology Rule**: 
- Ascending baseline = optimistic, positive
- Descending baseline = discouraged, negative
- Straight baseline = balanced, stable

---

## 5. Letter Size Extraction (`get_letter_size`)

**Purpose**: Measure average letter height to determine introversion/extroversion.

### Algorithm:

```python
def get_letter_size(img_or_path):
    # 1. Load original image (needs color info for proper thresholding)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. Multiple thresholding methods (fallback chain)
    # Method 1: Adaptive threshold
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 9)
    
    # Method 2: Otsu threshold (if adaptive fails)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Morphological operations to separate letters
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # Remove noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)   # Connect parts
    
    # 4. Find all contours (letters)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. Filter out huge contours (whole page) and noise
    max_contour_area = h * w * 0.95
    contours = [c for c in contours if cv2.contourArea(c) < max_contour_area]
    
    # 6. Measure letter heights with progressive filtering
    heights_mm = []
    px_per_mm = w / 210.0  # A4 width = 210mm
    
    for c in contours:
        x, y, w_box, h_box = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        
        # Filter: reasonable letter size (8-120px height, area > 20)
        if 8 < h_box < 120 and area > 20:
            h_mm = h_box / px_per_mm
            heights_mm.append(h_mm)
    
    # 7. Calculate median (robust to outliers)
    median_mm = np.median(heights_mm)
    
    # 8. Classify
    if median_mm <= 3.0:
        return median_mm, "Small Letter", "Introvert"
    else:
        return median_mm, "Large Letter", "Extrovert"
```

**Technologies**:
- `cv2.adaptiveThreshold`: Handles varying lighting
- `cv2.threshold` + `cv2.THRESH_OTSU`: Automatic threshold selection
- `cv2.morphologyEx`: Morphological operations (opening, closing)
- `cv2.findContours`: Find letter boundaries
- `cv2.boundingRect`: Get letter dimensions
- `np.median`: Robust central tendency

**Output**:
- `LetterSizeMM`: Numeric (millimeters)
- `LetterFeature`: Small Letter/Large Letter
- `LetterSizeTrait`: Introvert/Extrovert

**Graphology Rule**: 
- Small letters (< 3mm) = introverted, detail-oriented
- Large letters (> 3mm) = extroverted, expressive

---

## 6. Word Spacing Extraction (`get_word_spacing`)

**Purpose**: Measure spacing between words to determine social interaction style.

### Algorithm:

```python
def get_word_spacing(img):
    # 1. Morphological closing to connect letters into words
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 7))
    # Horizontal kernel (30 wide, 7 tall) connects letters horizontally
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    # 2. Find word contours (each word is now one blob)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Sort words by position (top to bottom, left to right)
    boxes = sorted([cv2.boundingRect(c) for c in contours], 
                   key=lambda b: (b[1], b[0]))  # Sort by y, then x
    
    # 4. Calculate gaps between words on same line
    gaps = []
    prev_y, prev_x, prev_w = None, None, None
    
    for (x, y, w, h) in boxes:
        if prev_y is None:
            prev_y, prev_x, prev_w = y, x, w
            continue
        
        # Check if words are on same line (y difference < 50px)
        if abs(y - prev_y) < 50:
            gap = x - (prev_x + prev_w)  # Gap = start of next - end of previous
            if 5 < gap < 300:  # Reasonable gap size
                gaps.append(gap)
        
        prev_y, prev_x, prev_w = y, x, w
    
    # 5. Calculate ratio: gap / word_width
    avg_gap = np.median(gaps)
    W_est = np.median([w for (_, _, w, _) in boxes]) * 1.4  # Estimated word width
    ratio = max(avg_gap / W_est, 0.1)
    
    # 6. Classify
    if ratio > 1.10:
        return ratio, "Wide", "Reserved"
    elif ratio < 0.85:
        return ratio, "Narrow", "Sociable"
    else:
        return ratio, "Balanced", "Balanced"
```

**Technologies**:
- `cv2.getStructuringElement`: Create morphological kernel
- `cv2.morphologyEx` + `MORPH_CLOSE`: Connect letters into words
- `cv2.boundingRect`: Get word bounding boxes
- `np.median`: Robust gap calculation

**Output**:
- `WordSpacingRatio`: Numeric ratio
- `WordFeature`: Narrow/Balanced/Wide
- Trait: Sociable/Balanced/Reserved

**Graphology Rule**: 
- Wide spacing = reserved, needs personal space
- Narrow spacing = sociable, comfortable with closeness

---

## 7. Line Spacing Extraction (`get_line_spacing`)

**Purpose**: Measure spacing between text lines to determine emotional space needs.

### Algorithm:

```python
def get_line_spacing(img):
    # 1. Vertical projection to find text lines
    proj = np.sum(img, axis=1)
    y_idx = np.where(proj > np.mean(proj) * 0.3)[0]
    
    # 2. Group consecutive rows into lines
    lines = []
    start = y_idx[0]
    for i in range(1, len(y_idx)):
        if y_idx[i] - y_idx[i - 1] > 5:
            lines.append((start, y_idx[i - 1]))
            start = y_idx[i]
    lines.append((start, y_idx[-1]))
    
    # 3. Calculate distances between lines
    distances = [lines[i + 1][0] - lines[i][1] for i in range(len(lines) - 1)]
    avg_space = np.median(distances)
    
    # 4. Calculate x-height (typical letter height)
    xheight = np.median([y2 - y1 for (y1, y2) in lines])
    
    # 5. Calculate ratio: spacing / x-height
    ratio = max(avg_space / xheight, 0.5)
    
    # 6. Classify
    if ratio > 1.2:
        return ratio, "Wide", "Distant"
    elif ratio < 0.8:
        return ratio, "Narrow", "Clingy"
    else:
        return ratio, "Moderate", "Balanced"
```

**Technologies**:
- `np.sum(axis=1)`: Vertical projection
- `np.median`: Robust statistics

**Output**:
- `LineSpacingRatio`: Numeric ratio
- `LineFeature`: Narrow/Moderate/Wide
- Trait: Clingy/Balanced/Distant

**Graphology Rule**: 
- Wide line spacing = needs emotional distance
- Narrow line spacing = emotionally clingy

---

## 8. Final Feature Combination (`extract_all_features`)

**Purpose**: Orchestrates all feature extractions and derives final personality traits.

### Process:

```python
def extract_all_features(img_path):
    # 1. Preprocess image
    img = preprocess_image(img_path)
    
    # 2. Extract all features
    data["SlantAngle"], data["SlantFeature"], data["EmotionalScale"] = get_slant_angle(img)
    (data["LeftMarginIn"], data["LeftMarginType"], left_trait,
     data["TopMarginIn"], data["TopMarginType"], top_trait) = get_margins(img_path)
    
    # 3. Derive Orientation from margins
    if left_trait == "Attached" and top_trait == "Ambitious":
        data["Orientation"] = "Goal-Oriented"
    # ... other combinations
    
    data["BaselineAngle"], data["BaselineFeature"], data["EmotionalOutlook"] = get_baseline_angle(img)
    data["LetterSizeMM"], data["LetterFeature"], data["LetterSizeTrait"] = get_letter_size(img_path)
    data["WordSpacingRatio"], data["WordFeature"], word_trait = get_word_spacing(img)
    data["LineSpacingRatio"], data["LineFeature"], line_trait = get_line_spacing(img)
    
    # 4. Derive SocialIsolation from word + line spacing
    if word_trait == "Reserved" and line_trait == "Balanced":
        data["SocialIsolation"] = "Reserved"
    # ... other combinations
    
    # 5. Derive Concentration from baseline + letter size + line spacing
    even_baseline = abs(data["BaselineAngle"]) < 2
    letter_size = data["LetterSizeMM"]
    ratio = data["LineSpacingRatio"]
    
    if letter_size <= 3 and even_baseline:
        data["Concentration"] = "Focused"
    elif letter_size <= 3 and not even_baseline:
        data["Concentration"] = "Tense"
    # ... other rules
    
    return pd.DataFrame([data])
```

**Output Features** (14 total):
1. `SlantAngle` (numeric)
2. `SlantFeature` (categorical)
3. `LeftMarginIn` (numeric)
4. `LeftMarginType` (categorical)
5. `TopMarginIn` (numeric)
6. `TopMarginType` (categorical)
7. `BaselineAngle` (numeric)
8. `BaselineFeature` (categorical)
9. `LetterSizeMM` (numeric)
10. `LetterFeature` (categorical)
11. `WordSpacingRatio` (numeric)
12. `WordFeature` (categorical)
13. `LineSpacingRatio` (numeric)
14. `LineFeature` (categorical)

**Output Traits** (6 total):
1. `EmotionalScale` (Suppressed/Balanced/Expressive/Unstable)
2. `Orientation` (Goal-Oriented/Stable/Independent Thinker/Reserved/Balanced)
3. `EmotionalOutlook` (Optimistic/Discouraged/Balanced)
4. `LetterSizeTrait` (Introvert/Extrovert)
5. `SocialIsolation` (Reserved/Balanced/Clingy/Sociable)
6. `Concentration` (Focused/Relaxed/Distracted/Tense)

---

## Key Computer Vision Techniques

### 1. **Edge Detection**
- **Canny Edge Detector**: Detects edges in images
- Used in: Slant angle detection

### 2. **Line Detection**
- **Hough Transform**: Detects lines in edge images
- Used in: Slant angle measurement

### 3. **Contour Detection**
- **cv2.findContours**: Finds object boundaries
- Used in: Letter detection, word detection, margin detection

### 4. **Morphological Operations**
- **Opening**: Removes noise (erosion + dilation)
- **Closing**: Fills gaps (dilation + erosion)
- Used in: Letter size, word spacing

### 5. **Thresholding**
- **Adaptive Threshold**: Handles varying lighting
- **Otsu Threshold**: Automatic threshold selection
- Used in: Image preprocessing, letter size

### 6. **Projection**
- **Vertical Projection**: Sum pixels in each row
- Used in: Line detection, baseline angle

### 7. **Linear Regression**
- **scipy.stats.linregress**: Fits line to points
- Used in: Baseline angle calculation

### 8. **Geometric Transformations**
- **Rotation**: Deskewing
- **Warping**: Image correction
- Used in: Image preprocessing

---

## Measurement Units

- **Angles**: Degrees (°)
- **Margins**: Inches (in)
- **Letter Size**: Millimeters (mm)
- **Spacing**: Ratios (dimensionless)

---

## Error Handling

The code includes multiple fallback mechanisms:

1. **Multiple thresholding methods**: If adaptive fails, try Otsu, then simple threshold
2. **Progressive filtering**: Relax filters if no letters found
3. **Default values**: Return safe defaults if extraction fails
4. **Contour filtering**: Filter out noise and whole-page contours

---

## Summary

The feature extraction system uses **computer vision** and **image processing** techniques to:
1. Preprocess images (deskew, binarize)
2. Extract 6 physical measurements (angles, sizes, spacing)
3. Classify into categorical features
4. Map to personality traits using graphology rules

All measurements are converted to standardized units and combined into a comprehensive feature vector for machine learning models.

