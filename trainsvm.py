# ============================================================
# STEP 4 — HYBRID CNN + SVM MODEL TRAINING (Concentration Prediction, Balanced)
# ============================================================

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import resnet
import joblib

# ---------------- PATHS ----------------
BASE_DIR = r"/Users/gagan/Desktop/graph"
CSV_PATH = os.path.join(BASE_DIR, "features_mapped1.csv")
IMG_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(BASE_DIR, "cnn_personality_model_resnet50.h5")
SAVE_DIR = os.path.join(BASE_DIR, "Hybrid_SVM_Step4_Balanced")
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- PARAMETERS ----------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ---------------- LOAD TRAINED CNN ----------------
print("📥 Loading trained CNN model...")
cnn_model = load_model(MODEL_PATH)
embedding_model = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-3].output)
print("✅ Embedding extractor ready (256-D output).")

# ---------------- LOAD DATA ----------------
df = pd.read_csv(CSV_PATH)
df = df.rename(columns={"ImageName": "filename"})
print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")

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

# ---------------- CLASS DISTRIBUTION ----------------
print("\n📊 Class Distribution:")
print(df[target_col].value_counts())

# ---------------- ENCODE CATEGORICAL FEATURES ----------------
print("\n🔠 Encoding categorical features...")
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_encoded = encoder.fit_transform(df[categorical_features])
print(f"✅ One-hot encoded categorical shape: {cat_encoded.shape}")

# Combine handcrafted numeric + encoded categorical
X_handcrafted = np.concatenate([df[numeric_features].values, cat_encoded], axis=1)
print(f"✅ Handcrafted features shape: {X_handcrafted.shape}")

# ---------------- ENCODE TARGET ----------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df[target_col])
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
print(f"✅ CNN embeddings shape: {cnn_features.shape}")

# ---------------- COMBINE CNN + HANDCRAFTED ----------------
X_combined = np.concatenate([cnn_features, X_handcrafted], axis=1)
print(f"✅ Final combined feature shape: {X_combined.shape}")

# ---------------- TRAIN/TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)

# ---------------- CLEAN INVALID VALUES ----------------
print("\n🧹 Cleaning NaN/Inf values before scaling...")
X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
if np.isfinite(X_train).all() and np.isfinite(X_test).all():
    print("✅ All numeric values are finite — safe to scale.")
else:
    raise ValueError("⚠️ Still contains invalid values after cleaning!")

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))
joblib.dump(encoder, os.path.join(SAVE_DIR, "encoder.pkl"))
joblib.dump(label_encoder, os.path.join(SAVE_DIR, "label_encoder.pkl"))
print("✅ Scaler & encoders saved.")

# ---------------- TRAIN HYBRID SVM (BALANCED) ----------------
print("\n🚀 Training Hybrid CNN + SVM classifier (balanced)...")
svm = CalibratedClassifierCV(SVC(kernel='rbf', C=1, probability=True, class_weight='balanced'))
svm.fit(X_train_scaled, y_train)

# ---------------- EVALUATION ----------------
print("\n📈 Evaluating model...")
y_pred = svm.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Hybrid CNN+SVM Accuracy (Balanced): {acc*100:.2f}%")

print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------- SAVE MODEL ----------------
model_path = os.path.join(SAVE_DIR, "hybrid_cnn_svm_concentration_balanced.pkl")
joblib.dump(svm, model_path)
print(f"\n💾 Model saved to: {model_path}")

print("\n✅ STEP 4 — Balanced Hybrid CNN + SVM Training Completed Successfully!")
