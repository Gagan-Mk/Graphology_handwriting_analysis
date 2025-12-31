import os
import glob
import argparse
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet import preprocess_input
import joblib

# Reuse the existing feature extraction routine without changing the original file
from feature_extaraction import extract_all_features


def collect_images(folder: str) -> list[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.heif", "*.heic")
    image_paths: list[str] = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(folder, ext)))
    return image_paths


def ensure_order(df: pd.DataFrame) -> pd.DataFrame:
    ordered_cols = [
        "ImageName",
        "SlantAngle", "SlantFeature", "EmotionalScale",
        "LeftMarginIn", "LeftMarginType", "TopMarginIn", "TopMarginType", "Orientation",
        "BaselineAngle", "BaselineFeature", "EmotionalOutlook",
        "LetterSizeMM", "LetterFeature", "LetterSizeTrait",
        "WordSpacingRatio", "WordFeature", "SocialIsolation",
        "LineSpacingRatio", "LineFeature", "Concentration",
    ]
    return df[ordered_cols]


def append_row(row_df: pd.DataFrame, csv_path: str) -> None:
    row_df = ensure_order(row_df)
    header = not os.path.exists(csv_path)
    row_df.to_csv(csv_path, mode="a", header=header, index=False)


def main() -> None:
    # Defaults tailored for your project layout on macOS
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_folder = os.path.join(base_dir, "results")
    default_output = os.path.join(base_dir, "features_mapped1.csv")
    cnn_model_path = os.path.join(base_dir, "cnn_personality_model_resnet50.h5")
    svm_path = os.path.join(base_dir, "Hybrid_SVM_Step4_Balanced", "hybrid_cnn_svm_concentration_balanced.pkl")
    scaler_path = os.path.join(base_dir, "Hybrid_SVM_Step4_Balanced", "scaler.pkl")
    encoder_path = os.path.join(base_dir, "Hybrid_SVM_Step4_Balanced", "encoder.pkl")
    label_encoder_path = os.path.join(base_dir, "Hybrid_SVM_Step4_Balanced", "label_encoder.pkl")

    parser = argparse.ArgumentParser(
        description="Updated feature extractor: single image (--image) or batch (--folder)."
    )
    parser.add_argument("--image", type=str, help="Absolute or relative path to a single image")
    parser.add_argument("--folder", type=str, help="Folder to batch process images", default=default_folder)
    parser.add_argument("--output", type=str, help="Output CSV path", default=default_output)
    args = parser.parse_args()

    output_csv = args.output if os.path.isabs(args.output) else os.path.abspath(args.output)

    # If no --image was supplied, ask interactively; empty input falls back to batch mode
    image_arg = args.image
    if image_arg is None:
        try:
            user_input = input(f"Enter image path for single-image extraction (or press Enter to batch '{default_folder}'): ").strip()
        except EOFError:
            user_input = ""
        if user_input:
            image_arg = user_input

    if image_arg:
        img_path = image_arg if os.path.isabs(image_arg) else os.path.abspath(image_arg)
        if not os.path.exists(img_path):
            print("❌ Image not found:", img_path)
            raise SystemExit(1)
        print(f"\n🖼️ Processing single image: {os.path.basename(img_path)}")
        try:
            row_df = extract_all_features(img_path)
            # Preview
            print(ensure_order(row_df).T)
            append_row(row_df, output_csv)
            print("\n✅ Appended features to:", output_csv)
            # ---- Prediction only for this image ----
            print("\n🤖 Predicting concentration for this image...")
            # Load models
            cnn_model = load_model(cnn_model_path)
            embedding_model = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-3].output)
            svm_model = joblib.load(svm_path)
            scaler = joblib.load(scaler_path)
            encoder = joblib.load(encoder_path)
            label_encoder = joblib.load(label_encoder_path)

            # CNN embedding
            IMG_SIZE = (224, 224)
            img_pil = keras_image.load_img(img_path, target_size=IMG_SIZE)
            img_arr = keras_image.img_to_array(img_pil)
            img_arr = np.expand_dims(img_arr, axis=0)
            img_arr = preprocess_input(img_arr)
            cnn_emb = embedding_model.predict(img_arr, verbose=0)

            # Handcrafted features
            numeric_features = [
                "SlantAngle", "LeftMarginIn", "TopMarginIn",
                "BaselineAngle", "LetterSizeMM", "WordSpacingRatio", "LineSpacingRatio"
            ]
            categorical_features = [
                "SlantFeature", "EmotionalScale", "LeftMarginType", "TopMarginType",
                "Orientation", "BaselineFeature", "EmotionalOutlook",
                "LetterFeature", "LetterSizeTrait", "WordFeature",
                "SocialIsolation", "LineFeature"
            ]
            num_vals = row_df[numeric_features].values
            cat_vals = encoder.transform(row_df[categorical_features])
            handcrafted = np.concatenate([num_vals, cat_vals], axis=1)
            combined = np.concatenate([cnn_emb, handcrafted], axis=1)
            combined_scaled = scaler.transform(combined)
            pred = svm_model.predict(combined_scaled)
            pred_label = label_encoder.inverse_transform(pred)[0]
            print(f"✅ Predicted Concentration: {pred_label}")
        except Exception as e:
            print(f"❌ Error {os.path.basename(img_path)}: {e}")
            raise SystemExit(1)
        return

    # Batch mode
    folder = args.folder if os.path.isabs(args.folder) else os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        print("❌ Folder not found:", folder)
        raise SystemExit(1)

    image_paths = collect_images(folder)
    if not image_paths:
        print("⚠️ No images found in:", folder)
        raise SystemExit(0)

    all_rows: list[pd.DataFrame] = []
    for img_path in image_paths:
        print(f"\n🖼️ Processing: {os.path.basename(img_path)}")
        try:
            row_df = extract_all_features(img_path)
            all_rows.append(row_df)
            print(ensure_order(row_df).T)
        except Exception as e:
            print(f"❌ Error {os.path.basename(img_path)}: {e}")

    if not all_rows:
        print("⚠️ No valid images processed.")
        raise SystemExit(1)

    final_df = ensure_order(pd.concat(all_rows, ignore_index=True))
    # If output exists, append without header; else write header
    if os.path.exists(output_csv):
        final_df.to_csv(output_csv, mode="a", header=False, index=False)
        print("\n✅ Appended", len(final_df), "rows to:", output_csv)
    else:
        final_df.to_csv(output_csv, index=False)
        print("\n✅ Saved features and traits to:", output_csv)


if __name__ == "__main__":
    main()


