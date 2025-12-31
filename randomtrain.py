# ============================================================
# RANDOM FOREST - Personality Trait Prediction (One Model + One CSV)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import os

# ---------------- CONFIG ----------------
CSV_PATH = r"C:\Users\venka\Desktop\graph\features_mapped1.csv"
MODEL_DIR = r"C:\Users\venka\Desktop\graph\rf_model_output"
RANDOM_STATE = 42
N_ESTIMATORS = 200
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(CSV_PATH)
print(f"✅ Dataset loaded: {df.shape}")

# ---------------- SELECT FEATURES ----------------
selected_features = [
    "SlantAngle", "LeftMarginIn", "TopMarginIn", "BaselineAngle", "LetterSizeMM",
    "WordSpacingRatio", "LineSpacingRatio", "SlantFeature", "LeftMarginType",
    "TopMarginType", "BaselineFeature", "LetterFeature", "WordFeature", "LineFeature"
]

targets = [
    "EmotionalScale", "EmotionalOutlook", "LetterSizeTrait",
    "SocialIsolation", "Concentration", "Orientation"
]

df = df[selected_features + targets]

# ---------------- SPLIT X / y ----------------
X = df[selected_features]
y = df[targets]

# ---------------- PREPROCESS ----------------
num_cols = ["SlantAngle", "LeftMarginIn", "TopMarginIn", "BaselineAngle",
            "LetterSizeMM", "WordSpacingRatio", "LineSpacingRatio"]
cat_cols = [c for c in selected_features if c not in num_cols]

print("\n🔍 Checking numeric values...")
for col in num_cols:
    X[col] = X[col].replace([np.inf, -np.inf], np.nan).fillna(X[col].median())

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=False)

# Scale numeric columns
scaler = StandardScaler()
X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
print("✅ Numeric data cleaned and scaled successfully.")

# Encode targets
label_encoders = {}
y_encoded = pd.DataFrame()
for col in targets:
    le = LabelEncoder()
    y_encoded[col] = le.fit_transform(y[col].astype(str))
    label_encoders[col] = le
    joblib.dump(le, os.path.join(MODEL_DIR, f"label_encoder_{col}.joblib"))
print("✅ Target encoding completed.")

# ---------------- TRAIN / TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=RANDOM_STATE
)
print(f"📊 Train: {X_train.shape}, Test: {X_test.shape}")

# ---------------- TRAIN MODEL ----------------
rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
multi_rf = MultiOutputClassifier(rf, n_jobs=-1)

print("🚀 Training Random Forest model...")
multi_rf.fit(X_train, y_train)

# Save only one combined model
model_path = os.path.join(MODEL_DIR, "multioutput_randomforest.joblib")
joblib.dump(multi_rf, model_path)
print(f"💾 Combined model saved to {model_path}")

# ---------------- EVALUATION ----------------
results = []
for i, col in enumerate(targets):
    y_pred = multi_rf.predict(X_test)[:, i]
    y_true = y_test.iloc[:, i].values
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    results.append((col, acc, f1))
    print(f"\n=== {col} ===")
    print(f"Accuracy: {acc:.3f} | F1-score: {f1:.3f}")
    print(classification_report(y_true, y_pred, zero_division=0))

perf_df = pd.DataFrame(results, columns=["Trait", "Accuracy", "F1_Score"])
perf_df.to_csv(os.path.join(MODEL_DIR, "model_performance.csv"), index=False)
print("\n📊 Model performance saved!")

# ---------------- COMBINED FEATURE IMPORTANCE ----------------
feature_names = X_encoded.columns.tolist()
importances_all = np.zeros(len(feature_names))

# Average feature importances from all estimators
for est in multi_rf.estimators_:
    importances_all += est.feature_importances_

importances_all /= len(multi_rf.estimators_)
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Average_Importance": importances_all
}).sort_values("Average_Importance", ascending=False)

importance_path = os.path.join(MODEL_DIR, "combined_feature_importance.csv")
importance_df.to_csv(importance_path, index=False)
print(f"💾 Combined feature importance saved to: {importance_path}")

# ---------------- SAVE FULL DATASET PREDICTIONS ----------------
y_pred_all = multi_rf.predict(X_encoded)
pred_df = pd.DataFrame(y_pred_all, columns=targets)

# Decode back to readable labels
for col in targets:
    pred_df[col] = label_encoders[col].inverse_transform(pred_df[col])

output_full = pd.concat([df, pred_df.add_suffix("_Predicted")], axis=1)
output_path = os.path.join(MODEL_DIR, "full_dataset_predictions.csv")
output_full.to_csv(output_path, index=False)

print(f"\n✅ Full dataset predictions saved to: {output_path}")
print("\n🎉 Training completed successfully — one model + one feature CSV!")
