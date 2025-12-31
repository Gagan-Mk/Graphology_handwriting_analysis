# ============================================================
# TESTING SCRIPT — Image Path Input for Personality Prediction
# ============================================================

import pandas as pd
import numpy as np
import joblib
import os
import sys
from feature_extaraction import extract_all_features

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "rf_model_output")

# ---------------- LOAD COMPONENTS ----------------
print("🔄 Loading model and encoders...")
model = joblib.load(os.path.join(MODEL_DIR, "multioutput_randomforest.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))

targets = [
    "EmotionalScale", "EmotionalOutlook", "LetterSizeTrait",
    "SocialIsolation", "Concentration", "Orientation"
]
label_encoders = {t: joblib.load(os.path.join(MODEL_DIR, f"label_encoder_{t}.joblib")) for t in targets}
print("✅ Model and encoders loaded successfully.\n")

# ---------------- GET IMAGE PATH ----------------
if len(sys.argv) > 1:
    # Image path provided as command line argument
    img_path = sys.argv[1]
else:
    # Ask user for image path
    img_path = input("📸 Enter the path to the handwriting image: ").strip()

# Handle relative and absolute paths
if not os.path.isabs(img_path):
    img_path = os.path.abspath(img_path)

if not os.path.exists(img_path):
    print(f"❌ Error: Image not found at: {img_path}")
    sys.exit(1)

# ---------------- EXTRACT FEATURES FROM IMAGE ----------------
print(f"\n🖼️  Processing image: {os.path.basename(img_path)}")
print("🔍 Extracting handwriting features...")
try:
    features_df = extract_all_features(img_path)
    print("✅ Features extracted successfully!\n")
except Exception as e:
    print(f"❌ Error extracting features: {e}")
    sys.exit(1)

# ---------------- SELECT REQUIRED FEATURES ----------------
selected_features = [
    "SlantAngle", "LeftMarginIn", "TopMarginIn", "BaselineAngle", "LetterSizeMM",
    "WordSpacingRatio", "LineSpacingRatio", "SlantFeature", "LeftMarginType",
    "TopMarginType", "BaselineFeature", "LetterFeature", "WordFeature", "LineFeature"
]

# Extract only the features needed for prediction
sample = features_df[selected_features].copy()

# Display extracted features
print("📊 Extracted Features:")
print("-" * 50)
for feat in selected_features:
    if feat in sample.columns:
        print(f"{feat:<25}: {sample[feat].iloc[0]}")
print("-" * 50)
print()

# ---------------- PREPROCESS ----------------
num_cols = ["SlantAngle", "LeftMarginIn", "TopMarginIn", "BaselineAngle",
            "LetterSizeMM", "WordSpacingRatio", "LineSpacingRatio"]
cat_cols = [c for c in sample.columns if c not in num_cols]

# Handle numeric columns (replace inf/nan with median, same as training)
for col in num_cols:
    if col in sample.columns:
        sample[col] = sample[col].replace([np.inf, -np.inf], np.nan)
        if sample[col].isna().any():
            # Use a default median if all values are NaN
            median_val = sample[col].median() if not sample[col].isna().all() else 0.0
            sample[col] = sample[col].fillna(median_val)

X_encoded = pd.get_dummies(sample, columns=cat_cols, drop_first=False)

# Align with training feature columns
model_features = model.estimators_[0].feature_names_in_
for col in model_features:
    if col not in X_encoded.columns:
        X_encoded[col] = 0
X_encoded = X_encoded[model_features]

# Scale numeric columns
X_encoded[num_cols] = scaler.transform(X_encoded[num_cols])

# ---------------- PREDICT ----------------
y_pred = model.predict(X_encoded)[0]
decoded = {t: label_encoders[t].inverse_transform([y_pred[i]])[0] for i, t in enumerate(targets)}

# ---------------- DISPLAY RESULTS ----------------
print("\n🧠 Predicted Personality Traits:")
print("--------------------------------------------------")
for k, v in decoded.items():
    print(f"{k:<20}: {v}")
print("--------------------------------------------------")
