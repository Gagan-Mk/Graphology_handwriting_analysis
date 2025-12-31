# ============================================================
# STEP 5 — HYBRID CNN + SVM TESTING GUI (Multi-Upload Supported)
# ============================================================

import os
import numpy as np
import pandas as pd
import joblib
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Scrollbar, Canvas
from PIL import Image, ImageTk
import subprocess
import sys
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
from feature_extaraction import extract_all_features

# ---------------- PATHS ----------------
BASE_DIR = r"/Users/gagan/Desktop/graph"
CSV_PATH = os.path.join(BASE_DIR, "features_mapped1.csv")
IMG_DIR = os.path.join(BASE_DIR, "results")
CNN_MODEL_PATH = os.path.join(BASE_DIR, "cnn_personality_model_resnet50.h5")
HYBRID_MODEL_PATH = os.path.join(BASE_DIR, "Hybrid_SVM_Step4_Balanced", "hybrid_cnn_svm_concentration_balanced.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "Hybrid_SVM_Step4_Balanced", "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "Hybrid_SVM_Step4_Balanced", "encoder.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "Hybrid_SVM_Step4_Balanced", "label_encoder.pkl")

IMG_SIZE = (224, 224)

# ---------------- LOAD MODELS & DATA ----------------
print("📥 Loading CNN model and SVM model...")
cnn_model = load_model(CNN_MODEL_PATH)
embedding_model = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-3].output)

svm_model = joblib.load(HYBRID_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

df = pd.read_csv(CSV_PATH)
df = df.rename(columns={"ImageName": "filename"})

print("✅ All models and dataset loaded successfully!")

# ---------------- FEATURE DEFINITIONS ----------------
numeric_features = [
    'SlantAngle', 'LeftMarginIn', 'TopMarginIn',
    'BaselineAngle', 'LetterSizeMM', 'WordSpacingRatio', 'LineSpacingRatio'
]

categorical_features = [
    'SlantFeature', 'EmotionalScale', 'LeftMarginType', 'TopMarginType',
    'Orientation', 'BaselineFeature', 'EmotionalOutlook',
    'LetterFeature', 'LetterSizeTrait', 'WordFeature',
    'SocialIsolation', 'LineFeature'
]

# ---------------- PREDICTION FUNCTION ----------------
def predict_image(img_path, img_name):
    global df
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    cnn_emb = embedding_model.predict(img_array, verbose=0)

    row = df[df["filename"] == img_name]
    if row.empty:
        # Try calling the external updater script first
        try:
            updater_script = os.path.join(BASE_DIR, "feature_extaraction_updated.py")
            if os.path.exists(updater_script):
                proc = subprocess.run(
                    [sys.executable, updater_script, "--image", img_path, "--output", CSV_PATH],
                    capture_output=True,
                    text=True,
                    timeout=180
                )
                if proc.returncode != 0:
                    raise RuntimeError(proc.stderr or proc.stdout)
                # Reload CSV and try again
                df_local = pd.read_csv(CSV_PATH).rename(columns={"ImageName": "filename"})
                df[:] = df_local  # update in-memory df in place
                row = df[df["filename"] == img_name]
        except Exception:
            # Fallback: in-process feature extraction for unseen image
            try:
                fresh_row = extract_all_features(img_path)
                fresh_row["filename"] = img_name
                # Deduplicate any accidental duplicate columns to avoid pandas reindex errors
                fresh_row = fresh_row.loc[:, ~fresh_row.columns.duplicated()]
                # Append to in-memory df for subsequent queries in this session
                df = pd.concat([df, fresh_row.rename(columns={"ImageName": "filename"})], ignore_index=True)
                # Persist to CSV for future sessions
                header = not os.path.exists(CSV_PATH)
                # Ensure column order compatibility
                ordered_cols = [
                    "ImageName",
                    "SlantAngle", "SlantFeature", "EmotionalScale",
                    "LeftMarginIn", "LeftMarginType", "TopMarginIn", "TopMarginType", "Orientation",
                    "BaselineAngle", "BaselineFeature", "EmotionalOutlook",
                    "LetterSizeMM", "LetterFeature", "LetterSizeTrait",
                    "WordSpacingRatio", "WordFeature", "SocialIsolation",
                    "LineSpacingRatio", "LineFeature", "Concentration"
                ]
                cols_present = [c for c in ordered_cols if c in fresh_row.columns]
                to_save = fresh_row.loc[:, cols_present].copy()
                # Ensure ImageName matches the basename used for lookup (always set it)
                to_save["ImageName"] = img_name
                # Align to existing CSV header if the file already exists
                if os.path.exists(CSV_PATH):
                    existing_header = list(pd.read_csv(CSV_PATH, nrows=0).columns)
                    # Reindex to match existing header; add any missing columns as NaN
                    to_save = to_save.reindex(columns=existing_header, fill_value=pd.NA)
                    # Ensure the ImageName column preserved after reindex
                    to_save["ImageName"] = img_name
                    to_save.to_csv(CSV_PATH, mode="a", header=False, index=False)
                else:
                    # First-time write with header
                    to_save.to_csv(CSV_PATH, mode="w", header=True, index=False)

                # --- reload CSV into in-memory df so the GUI session sees the update immediately ---
                df_local = pd.read_csv(CSV_PATH).rename(columns={"ImageName": "filename"})
                df[:] = df_local  # update the global dataframe in-place
                row = df[df["filename"] == img_name]
            except Exception as e:
                return "Unknown", f"No matching CSV traits found and feature extraction failed: {e}"

    num_vals = row[numeric_features].values
    cat_vals = encoder.transform(row[categorical_features])
    handcrafted = np.concatenate([num_vals, cat_vals], axis=1)

    combined = np.concatenate([cnn_emb, handcrafted], axis=1)
    combined_scaled = scaler.transform(combined)
    pred = svm_model.predict(combined_scaled)
    pred_label = label_encoder.inverse_transform(pred)[0]

    r = row.iloc[0]
    traits = f"""
Slant angle: {r['SlantAngle']}° → Feature Type: {r['SlantFeature']} → Trait: {r['EmotionalScale']}
Top/Left margin: {r['TopMarginIn']} inch → Feature Type: {r['TopMarginType']} → Trait: {r['Orientation']}
Baseline angle: {r['BaselineAngle']}° → Feature Type: {r['BaselineFeature']} → Trait: {r['EmotionalOutlook']}
Letter Size: {r['LetterSizeMM']} mm → Feature Type: {r['LetterFeature']} → Trait: {r['LetterSizeTrait']}
Word Spacing Ratio: {r['WordSpacingRatio']} → Feature Type: {r['WordFeature']} → Trait: {r['SocialIsolation']}
Line Spacing: {r['LineSpacingRatio']}×x-height → Feature Type: {r['LineFeature']} → Trait: {r['Concentration']}

🧠 Final Personality Traits:
1. Emotional Scale → {r['EmotionalScale']}
2. Orientation → {r['Orientation']}
3. Emotional Outlook → {r['EmotionalOutlook']}
4. Letter Size → {r['LetterSizeTrait']}
5. Social Isolation → {r['SocialIsolation']}
6. Concentration (Predicted) → {pred_label}
"""
    return pred_label, traits

# ---------------- TKINTER GUI ----------------
root = tk.Tk()
root.title("🧠 Hybrid CNN + SVM Personality Prediction")
root.geometry("1200x850")
root.configure(bg="#eef2f3")

Label(root, text="Upload Handwriting Image(s) for Personality Analysis (Hybrid Model)",
      font=("Helvetica", 15, "bold"), bg="#eef2f3").pack(pady=10)

# --- Scrollable Frame for Results ---
canvas = Canvas(root, bg="#eef2f3", highlightthickness=0)
scroll_y = Scrollbar(root, orient="vertical", command=canvas.yview)
scroll_frame = Frame(canvas, bg="#eef2f3")

scroll_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
canvas.configure(yscrollcommand=scroll_y.set)
canvas.pack(fill="both", expand=True, side="left", padx=10)
scroll_y.pack(fill="y", side="right")

# --- Functions ---
def clear_results():
    """Clears previously displayed results to allow re-upload."""
    for widget in scroll_frame.winfo_children():
        widget.destroy()
    result_label.config(text="")

def upload_images():
    clear_results()
    file_paths = filedialog.askopenfilenames(title="Select Handwriting Images",
                                             filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_paths:
        return

    full_text = ""
    for img_path in file_paths:
        img_name = os.path.basename(img_path)
        pred_label, traits = predict_image(img_path, img_name)

        # Display image + prediction
        img = Image.open(img_path).resize((180, 180))
        img_tk = ImageTk.PhotoImage(img)

        img_frame = Frame(scroll_frame, bg="white", bd=2, relief="groove")
        img_frame.pack(padx=10, pady=10, fill="x")

        Label(img_frame, image=img_tk, bg="white").pack(side="left", padx=10, pady=10)
        Label(
            img_frame,
            text=f"🖼 {img_name}\nPredicted Concentration: {pred_label}\n\n{traits}",
            font=("Menlo", 11),
            justify="left",
            bg="white",
            fg="#111111",
            anchor="w",
            wraplength=700
        ).pack(side="left", padx=10, fill="x", expand=True)
        img_frame.image = img_tk  # Prevent garbage collection

    result_label.config(text="✅ Prediction Completed. Scroll to view results.")

# --- Buttons ---
Button(root, text="📂 Upload Images", command=upload_images,
       bg="#4a90e2", fg="white", font=("Helvetica", 12, "bold"),
       padx=15, pady=8, relief="raised").pack(pady=8)

Button(root, text="🧹 Clear Results", command=clear_results,
       bg="#e74c3c", fg="white", font=("Helvetica", 12, "bold"),
       padx=15, pady=8, relief="raised").pack(pady=5)

result_label = Label(root, text="", font=("Helvetica", 11, "italic"),
                     bg="#eef2f3", fg="#333")
result_label.pack(pady=8)

root.mainloop()
