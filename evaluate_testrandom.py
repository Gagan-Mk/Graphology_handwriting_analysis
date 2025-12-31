# ============================================================
# EVALUATION SCRIPT — Accuracy Evaluation for testrandom.py Model
# ============================================================

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "features_mapped1.csv")
MODEL_DIR = os.path.join(BASE_DIR, "rf_model_output")
RANDOM_STATE = 42

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

# ---------------- LOAD DATA ----------------
print("📂 Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"✅ Dataset loaded: {df.shape}")

# ---------------- SELECT FEATURES ----------------
selected_features = [
    "SlantAngle", "LeftMarginIn", "TopMarginIn", "BaselineAngle", "LetterSizeMM",
    "WordSpacingRatio", "LineSpacingRatio", "SlantFeature", "LeftMarginType",
    "TopMarginType", "BaselineFeature", "LetterFeature", "WordFeature", "LineFeature"
]

# Ensure all required columns exist
missing_cols = [col for col in selected_features + targets if col not in df.columns]
if missing_cols:
    print(f"⚠️  Warning: Missing columns: {missing_cols}")
    df = df[[col for col in selected_features + targets if col in df.columns]]
else:
    df = df[selected_features + targets]

# Remove rows with missing target values
df = df.dropna(subset=targets).reset_index(drop=True)
print(f"✅ Cleaned dataset: {df.shape}")

# ---------------- CHECK FOR DATA LEAKAGE / DETERMINISTIC FEATURES ----------------
print("\n🔍 Checking for potential data leakage or deterministic features...")
print("-"*70)

# Check if any features are perfectly correlated with targets (data leakage)
for trait in targets:
    for feat in selected_features:
        if feat in df.columns and trait in df.columns:
            # For categorical features, check if they're identical
            if df[feat].dtype == 'object' and df[trait].dtype == 'object':
                if df[feat].equals(df[trait]):
                    print(f"⚠️  WARNING: {feat} is identical to {trait} - potential data leakage!")
            # For numeric, check correlation
            elif df[feat].dtype in ['float64', 'int64'] and df[trait].dtype == 'object':
                # Group by trait and check if feature values are unique per trait
                grouped = df.groupby(trait)[feat].nunique()
                if len(grouped[grouped == 1]) > 0:
                    print(f"⚠️  WARNING: {feat} may be deterministic for {trait}")

# Check feature uniqueness
print(f"\n📊 Feature Statistics:")
print(f"   Total samples: {len(df)}")
print(f"   Unique combinations of input features: {df[selected_features].drop_duplicates().shape[0]}")
if df[selected_features].drop_duplicates().shape[0] < len(df) * 0.1:
    print("   ⚠️  WARNING: Very few unique feature combinations - model may be memorizing!")
print()

# ---------------- SPLIT X / y ----------------
X = df[selected_features].copy()
y = df[targets].copy()

# ---------------- PREPROCESS (Same as testrandom.py) ----------------
print("\n🔧 Preprocessing data...")
num_cols = ["SlantAngle", "LeftMarginIn", "TopMarginIn", "BaselineAngle",
            "LetterSizeMM", "WordSpacingRatio", "LineSpacingRatio"]
cat_cols = [c for c in selected_features if c not in num_cols]

# Handle numeric columns (same as training)
for col in num_cols:
    X[col] = X[col].replace([np.inf, -np.inf], np.nan).fillna(X[col].median())

# One-hot encode categorical features (same as testrandom.py)
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=False)

# Align with training feature columns (same as testrandom.py)
model_features = model.estimators_[0].feature_names_in_
for col in model_features:
    if col not in X_encoded.columns:
        X_encoded[col] = 0
X_encoded = X_encoded[model_features]

# Scale numeric columns (same as testrandom.py)
X_encoded[num_cols] = scaler.transform(X_encoded[num_cols])
print("✅ Preprocessing completed.\n")

# ---------------- ENCODE TARGETS ----------------
y_encoded = pd.DataFrame()
for col in targets:
    y_encoded[col] = label_encoders[col].transform(y[col].astype(str))

# ---------------- TRAIN/TEST SPLIT (Same as training) ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=RANDOM_STATE
)
print(f"📊 Train set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples\n")

# ---------------- CHECK FOR OVERFITTING ----------------
print("="*70)
print("CHECKING FOR OVERFITTING (Train vs Test Performance)")
print("="*70)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("\n⚠️  Train vs Test Accuracy Comparison:")
print("-"*70)
train_results = []
for i, trait in enumerate(targets):
    y_true_train = y_train.iloc[:, i].values
    y_pred_train_trait = y_pred_train[:, i]
    train_acc = accuracy_score(y_true_train, y_pred_train_trait)
    
    y_true_test = y_test.iloc[:, i].values
    y_pred_test_trait = y_pred_test[:, i]
    test_acc = accuracy_score(y_true_test, y_pred_test_trait)
    
    diff = train_acc - test_acc
    train_results.append({
        "Trait": trait,
        "Train_Accuracy": train_acc,
        "Test_Accuracy": test_acc,
        "Difference": diff
    })
    
    status = "⚠️  OVERFITTING" if diff > 0.05 else "✅ OK"
    print(f"{trait:<20}: Train={train_acc:.4f}, Test={test_acc:.4f}, Diff={diff:.4f} {status}")

# ---------------- EVALUATE ON TEST SET ----------------
print("\n" + "="*70)
print("EVALUATING MODEL ON TEST SET")
print("="*70)

# Per-trait metrics
results = []
print("\n📈 Per-Trait Performance Metrics:")
print("-"*70)
for i, trait in enumerate(targets):
    y_true = y_test.iloc[:, i].values
    y_pred = y_pred_test[:, i]
    
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    results.append({
        "Trait": trait,
        "Accuracy": acc,
        "F1_Macro": f1_macro,
        "F1_Weighted": f1_weighted
    })
    
    print(f"\n🎯 {trait}:")
    print(f"   Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"   F1-Score (Macro): {f1_macro:.4f}")
    print(f"   F1-Score (Weighted): {f1_weighted:.4f}")
    
    # Classification report - only show classes present in test set
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    label_names = [label_encoders[trait].classes_[i] for i in unique_labels if i < len(label_encoders[trait].classes_)]
    
    print(f"\n   Classification Report:")
    print(classification_report(
        y_true, y_pred,
        labels=unique_labels,
        target_names=label_names,
        zero_division=0
    ))

# Overall metrics
print("\n" + "="*70)
print("OVERALL PERFORMANCE METRICS")
print("="*70)

# Average accuracy across all traits
avg_accuracy = np.mean([r["Accuracy"] for r in results])
print(f"\n📊 Average Accuracy (across all traits): {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")

# Exact match accuracy (all traits correct for a sample)
exact_matches = np.all(y_pred_test == y_test.values, axis=1)
exact_match_acc = np.mean(exact_matches)
print(f"🎯 Exact Match Accuracy (all traits correct): {exact_match_acc:.4f} ({exact_match_acc*100:.2f}%)")
print(f"   ({np.sum(exact_matches)} out of {len(exact_matches)} samples)")

# Per-sample accuracy (average of correct traits per sample)
per_sample_acc = np.mean(y_pred_test == y_test.values, axis=1)
avg_per_sample_acc = np.mean(per_sample_acc)
print(f"📈 Average Per-Sample Accuracy: {avg_per_sample_acc:.4f} ({avg_per_sample_acc*100:.2f}%)")

# ---------------- CROSS-VALIDATION (Better Generalization Estimate) ----------------
print("\n" + "="*70)
print("CROSS-VALIDATION EVALUATION (5-Fold)")
print("="*70)
print("\n⚠️  Running cross-validation to better assess generalization...")
print("    This may take a few minutes...\n")

# Use the full dataset for CV
cv_scores = {}
for i, trait in enumerate(targets):
    y_trait = y_encoded.iloc[:, i].values
    
    # Use StratifiedKFold for better class distribution
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Create a wrapper to extract single trait prediction
    def predict_trait(X):
        pred_all = model.predict(X)
        return pred_all[:, i]
    
    # Manual CV since MultiOutputClassifier doesn't have direct CV support
    cv_accuracies = []
    for train_idx, val_idx in cv.split(X_encoded, y_trait):
        X_cv_train, X_cv_val = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
        y_cv_train, y_cv_val = y_trait[train_idx], y_trait[val_idx]
        
        # Train a temporary model for this fold
        rf_temp = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        rf_temp.fit(X_cv_train, y_cv_train)
        y_pred_cv = rf_temp.predict(X_cv_val)
        cv_acc = accuracy_score(y_cv_val, y_pred_cv)
        cv_accuracies.append(cv_acc)
    
    cv_mean = np.mean(cv_accuracies)
    cv_std = np.std(cv_accuracies)
    cv_scores[trait] = {"mean": cv_mean, "std": cv_std}
    
    print(f"{trait:<20}: {cv_mean:.4f} ± {cv_std:.4f}")

print("\n📊 Cross-Validation Summary:")
print("-"*70)
cv_summary = pd.DataFrame({
    "Trait": list(cv_scores.keys()),
    "CV_Mean_Accuracy": [cv_scores[t]["mean"] for t in cv_scores.keys()],
    "CV_Std": [cv_scores[t]["std"] for t in cv_scores.keys()],
    "Test_Accuracy": [r["Accuracy"] for r in results]
})
print(cv_summary.to_string(index=False))

# Compare CV vs Test
print("\n⚠️  Overfitting Check (CV vs Test):")
print("-"*70)
for trait in targets:
    cv_acc = cv_scores[trait]["mean"]
    test_acc = next(r["Accuracy"] for r in results if r["Trait"] == trait)
    diff = test_acc - cv_acc
    status = "⚠️  OVERFITTING" if diff > 0.05 else "✅ OK"
    print(f"{trait:<20}: CV={cv_acc:.4f}, Test={test_acc:.4f}, Diff={diff:.4f} {status}")

# ---------------- SAVE RESULTS ----------------
results_df = pd.DataFrame(results)
results_df = results_df.merge(
    pd.DataFrame({
        "Trait": [r["Trait"] for r in train_results],
        "Train_Accuracy": [r["Train_Accuracy"] for r in train_results],
        "CV_Mean_Accuracy": [cv_scores[t]["mean"] for t in targets],
        "CV_Std": [cv_scores[t]["std"] for t in targets]
    }),
    on="Trait"
)
results_path = os.path.join(MODEL_DIR, "evaluation_results_testrandom.csv")
results_df.to_csv(results_path, index=False)
print(f"\n💾 Detailed results saved to: {results_path}")

# ---------------- EVALUATE ON FULL DATASET (Optional) ----------------
print("\n" + "="*70)
print("EVALUATING ON FULL DATASET")
print("="*70)

y_pred_full = model.predict(X_encoded)

results_full = []
for i, trait in enumerate(targets):
    y_true = y_encoded.iloc[:, i].values
    y_pred = y_pred_full[:, i]
    
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    results_full.append({
        "Trait": trait,
        "Accuracy": acc,
        "F1_Macro": f1_macro
    })

avg_accuracy_full = np.mean([r["Accuracy"] for r in results_full])
exact_matches_full = np.all(y_pred_full == y_encoded.values, axis=1)
exact_match_acc_full = np.mean(exact_matches_full)

print(f"\n📊 Full Dataset - Average Accuracy: {avg_accuracy_full:.4f} ({avg_accuracy_full*100:.2f}%)")
print(f"🎯 Full Dataset - Exact Match Accuracy: {exact_match_acc_full:.4f} ({exact_match_acc_full*100:.2f}%)")
print(f"   ({np.sum(exact_matches_full)} out of {len(exact_matches_full)} samples)")

# ---------------- SUMMARY TABLE ----------------
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
summary_df = pd.DataFrame({
    "Trait": [r["Trait"] for r in results],
    "Test_Accuracy": [r["Accuracy"] for r in results],
    "Test_F1_Macro": [r["F1_Macro"] for r in results],
    "Full_Accuracy": [r["Accuracy"] for r in results_full],
    "Full_F1_Macro": [r["F1_Macro"] for r in results_full]
})
print(summary_df.to_string(index=False))

print("\n✅ Evaluation completed!")

