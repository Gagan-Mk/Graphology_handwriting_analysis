# ============================================================
# EVALUATE HYBRID CNN + SVM MODEL ACCURACY
# ============================================================

import os
import glob
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import resnet
from sklearn.model_selection import train_test_split

# ---------------- PATHS ----------------
BASE_DIR = r"/Users/gagan/Desktop/graph"
CSV_PATH = os.path.join(BASE_DIR, "features_mapped1.csv")
IMG_DIR = os.path.join(BASE_DIR, "results")
CNN_MODEL_PATH = os.path.join(BASE_DIR, "cnn_personality_model_resnet50.h5")
SAVE_DIR = os.path.join(BASE_DIR, "Hybrid_SVM_Step4_Balanced")

MODEL_PATH = os.path.join(SAVE_DIR, "hybrid_cnn_svm_concentration_balanced.pkl")
SCALER_PATH = os.path.join(SAVE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(SAVE_DIR, "encoder.pkl")
LABEL_ENCODER_PATH = os.path.join(SAVE_DIR, "label_encoder.pkl")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ---------------- LOAD MODELS ----------------
print("📥 Loading models...")
cnn_model = load_model(CNN_MODEL_PATH)
embedding_model = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-3].output)
print("✅ CNN embedding extractor loaded.")

svm_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
print("✅ SVM model, scaler, and encoders loaded.")

# ---------------- LOAD DATA ----------------
df = pd.read_csv(CSV_PATH)
df = df.rename(columns={"ImageName": "filename"})
print(f"✅ Dataset loaded: {df.shape[0]} samples")

# Filter to only include images that actually exist
image_files = (glob.glob(os.path.join(IMG_DIR, "*.png")) + 
               glob.glob(os.path.join(IMG_DIR, "*.jpg")) + 
               glob.glob(os.path.join(IMG_DIR, "*.jpeg")))
existing_images = set([os.path.basename(f) for f in image_files])
df = df[df["filename"].isin(existing_images)].reset_index(drop=True)
print(f"✅ Filtered to {df.shape[0]} samples with existing images")

# ---------------- SELECT FEATURE COLUMNS ----------------
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

target_col = 'Concentration'

# ---------------- PREPARE FEATURES ----------------
print("\n🔠 Encoding features...")
cat_encoded = encoder.transform(df[categorical_features])
X_handcrafted = np.concatenate([df[numeric_features].values, cat_encoded], axis=1)

# ---------------- ENCODE TARGET ----------------
y_encoded = label_encoder.transform(df[target_col])
print(f"✅ Target classes: {list(label_encoder.classes_)}")

# ---------------- EXTRACT CNN EMBEDDINGS ----------------
print("\n🧠 Extracting CNN embeddings (this may take a few minutes)...")
datagen = ImageDataGenerator(preprocessing_function=resnet.preprocess_input)
gen = datagen.flow_from_dataframe(
    df,
    directory=IMG_DIR,
    x_col="filename",
    y_col=target_col,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)
cnn_features = embedding_model.predict(gen, verbose=1)

# ---------------- COMBINE FEATURES ----------------
X_combined = np.concatenate([cnn_features, X_handcrafted], axis=1)

# ---------------- TRAIN/TEST SPLIT (same as training) ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)

# ---------------- CLEAN AND SCALE ----------------
X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- EVALUATE ON TEST SET ----------------
print("\n📈 Evaluating model on test set...")
y_pred_test = svm_model.predict(X_test_scaled)
acc_test = accuracy_score(y_test, y_pred_test)

print(f"\n🎯 Test Set Accuracy: {acc_test*100:.2f}%")
print(f"📊 Test Set Size: {len(y_test)} samples")

print("\n" + "="*60)
print("CLASSIFICATION REPORT (Test Set):")
print("="*60)
print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

print("\n" + "="*60)
print("CONFUSION MATRIX (Test Set):")
print("="*60)
cm = confusion_matrix(y_test, y_pred_test)
print(cm)
print(f"\nClass labels: {list(label_encoder.classes_)}")

# ---------------- EVALUATE ON FULL DATASET ----------------
print("\n" + "="*60)
print("EVALUATING ON FULL DATASET:")
print("="*60)
X_full_scaled = scaler.transform(np.nan_to_num(X_combined, nan=0.0, posinf=1e6, neginf=-1e6))
y_pred_full = svm_model.predict(X_full_scaled)
acc_full = accuracy_score(y_encoded, y_pred_full)

print(f"\n🎯 Full Dataset Accuracy: {acc_full*100:.2f}%")
print(f"📊 Full Dataset Size: {len(y_encoded)} samples")

print("\n" + "="*60)
print("CLASSIFICATION REPORT (Full Dataset):")
print("="*60)
print(classification_report(y_encoded, y_pred_full, target_names=label_encoder.classes_))

# ---------------- CLASS DISTRIBUTION ----------------
print("\n" + "="*60)
print("CLASS DISTRIBUTION:")
print("="*60)
class_counts = pd.Series(label_encoder.inverse_transform(y_encoded)).value_counts()
print(class_counts)

print("\n✅ Evaluation completed!")

