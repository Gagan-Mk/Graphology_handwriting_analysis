# ============================================================
# CNN MODEL TRAINING — USING RESNET50 (Improved for High Accuracy)
# ============================================================

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, resnet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# ---------------- PATHS ----------------
BASE_DIR = r"C:\Users\venka\Desktop\graph"
CSV_PATH = os.path.join(BASE_DIR, "features_mapped1.csv")
IMG_DIR = os.path.join(BASE_DIR, "results")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "cnn_personality_model_resnet50.h5")
WEIGHTS_PATH = os.path.join(BASE_DIR, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

# ---------------- PARAMETERS ----------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10    # Train top layers first
EPOCHS_PHASE2 = 50    # Fine-tune last ResNet50 layers
LEARNING_RATE_PHASE1 = 1e-4
LEARNING_RATE_PHASE2 = 1e-5

# ---------------- LOAD CSV ----------------
df = pd.read_csv(CSV_PATH)
df = df.rename(columns={"ImageName": "filename", "Concentration": "label"})

# ---------------- DATA AUGMENTATION ----------------
datagen = ImageDataGenerator(
    preprocessing_function=resnet.preprocess_input,
    validation_split=0.25,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)

train_gen = datagen.flow_from_dataframe(
    df,
    directory=IMG_DIR,
    x_col="filename",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_dataframe(
    df,
    directory=IMG_DIR,
    x_col="filename",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# ---------------- MODEL ----------------
base_model = ResNet50(
    weights=WEIGHTS_PATH,
    include_top=False,
    input_shape=(224, 224, 3)
)

# ---------------- TOP LAYERS ----------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
preds = Dense(len(train_gen.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)

# ---------------- CALLBACKS ----------------
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-6, verbose=1)
]

# ---------------- CLASS WEIGHTS ----------------
label_to_index = train_gen.class_indices
class_weights_values = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df['label']),
    y=df['label']
)
class_weights = {label_to_index[label]: weight for label, weight in zip(np.unique(df['label']), class_weights_values)}
print("✅ Computed class weights:", class_weights)

# ---------------- PHASE 1: TRAIN TOP LAYERS ----------------
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_PHASE1),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("🚀 Phase 1: Training top layers...")
model.fit(
    train_gen,
    epochs=EPOCHS_PHASE1,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ---------------- PHASE 2: FINE-TUNE LAST RESNET50 LAYERS ----------------
for layer in base_model.layers[:-50]:
    layer.trainable = False
for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_PHASE2),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("🚀 Phase 2: Fine-tuning last ResNet50 layers...")
history = model.fit(
    train_gen,
    epochs=EPOCHS_PHASE2,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ---------------- SAVE MODEL ----------------
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved at: {MODEL_SAVE_PATH}")
