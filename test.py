import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input

# ---------------- PATHS ----------------
BASE_DIR = r"C:\Users\venka\Desktop\graph"
MODEL_PATH = os.path.join(BASE_DIR, "cnn_personality_model_resnet50.h5")
CSV_PATH = os.path.join(BASE_DIR, "features_mapped1.csv")
IMG_DIR = os.path.join(BASE_DIR, "results")

# ---------------- LOAD MODEL & DATA ----------------
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

df = pd.read_csv(CSV_PATH)
print("✅ Dataset loaded successfully!")

CLASS_LABELS = {i: label for i, label in enumerate(df["Concentration"].unique())}
IMG_SIZE = (224, 224)

# ---------------- PREDICTION FUNCTION ----------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    preds = model.predict(img_array)
    pred_label = CLASS_LABELS[np.argmax(preds)]
    return pred_label

# ---------------- DISPLAY LOGIC ----------------
def get_traits_from_csv(image_name):
    row = df[df["ImageName"] == image_name]
    if row.empty:
        return None

    r = row.iloc[0]
    traits = f"""
Slant angle: {r['SlantAngle']}° → Feature Type: {r['SlantFeature']} → Trait: {r['EmotionalScale']}
Top/Left margin: {r['TopMarginIn']} inch → Feature Type: {r['TopMarginType']} → Trait: {r['Orientation']}
Baseline angle: {r['BaselineAngle']}° → Feature Type: {r['BaselineFeature']} → Trait: {r['EmotionalOutlook']}
Letter Size: {r['LetterSizeMM']} mm → Feature Type: {r['LetterFeature']} → Trait: {r['LetterSizeTrait']}
Word Spacing Ratio: {r['WordSpacingRatio']} → Feature Type: {r['WordFeature']} → Trait: {r['SocialIsolation']}
Line Spacing: {r['LineSpacingRatio']}×x-height → Feature Type: {r['LineFeature']} → Trait: {r['Concentration']}

Final Personality Traits:
1. Emotional Scale → {r['EmotionalScale']}
2. Orientation → {r['Orientation']}
3. Emotional Outlook → {r['EmotionalOutlook']}
4. Letter Size → {r['LetterSizeTrait']}
5. Social Isolation → {r['SocialIsolation']}
6. Concentration → {r['Concentration']}
"""
    return traits

# ---------------- TKINTER GUI ----------------
root = tk.Tk()
root.title("🧠 Personality Prediction — ResNet50 + Dataset Traits")
root.geometry("1100x800")
root.configure(bg="#eef2f3")

title_label = Label(root, text="Upload Handwriting Image(s) for Personality Analysis",
                    font=("Helvetica", 15, "bold"), bg="#eef2f3")
title_label.pack(pady=10)

frame = Frame(root, bg="#eef2f3")
frame.pack(pady=10)

result_label = Label(root, text="", font=("Consolas", 12), justify="left", bg="#eef2f3")
result_label.pack(pady=10)

def upload_images():
    file_paths = filedialog.askopenfilenames(title="Select Handwriting Images",
                                             filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_paths:
        return

    for widget in frame.winfo_children():
        widget.destroy()

    full_text = ""
    for img_path in file_paths:
        img_name = os.path.basename(img_path)
        pred_label = predict_image(img_path)
        traits = get_traits_from_csv(img_name)

        # Show image preview
        img = Image.open(img_path)
        img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        img_label = Label(frame, image=img_tk)
        img_label.image = img_tk
        img_label.pack(side="left", padx=10)

        # Combine results
        full_text += f"\n🖼 {img_name}\nPredicted Concentration: {pred_label}\n\n{traits}\n"

    result_label.config(text=full_text)

upload_btn = Button(root, text="📂 Upload Images", command=upload_images,
                    bg="#4a90e2", fg="white", font=("Helvetica", 12, "bold"),
                    padx=15, pady=8, relief="raised")
upload_btn.pack(pady=10)

root.mainloop()
