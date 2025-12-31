Below is a clean, conference-ready GitHub README you can directly paste into your repository.
It’s written to sound academic + practical, which suits both GitHub reviewers and paper reviewers.

⸻

🧠 Handwriting-Based Personality Trait Analysis

Hybrid CNN + Graphology Feature Extraction Approach

📌 Project Overview

This project presents a hybrid personality prediction system based on handwritten text analysis, combining:
	•	Deep learning (CNN – ResNet50) for visual feature extraction
	•	Handcrafted graphological features derived using computer vision
	•	SVM classifier for robust personality trait prediction

The system aims to bridge traditional graphology principles with modern machine learning, enabling interpretable and data-driven personality assessment from handwriting samples.

⸻

🎯 Objectives
	•	Extract meaningful graphological features from handwriting images
	•	Learn deep visual representations using transfer learning
	•	Combine both feature types into a hybrid feature vector
	•	Predict multiple personality traits with improved accuracy and interpretability

⸻

🧩 System Architecture

Handwriting Image
        ↓
Image Preprocessing (Deskew, Thresholding)
        ↓
┌───────────────────────────────┐
│ Parallel Feature Extraction   │
├───────────────────────────────┤
│ • CNN Embeddings (ResNet50)   │ → 256-D
│ • Graphology Features        │ → 14 Features
└───────────────────────────────┘
        ↓
Feature Concatenation (Hybrid Vector)
        ↓
Scaling & Normalization
        ↓
SVM Classifier
        ↓
Personality Trait Prediction


⸻

🛠️ Technologies Used
	•	Python
	•	OpenCV – image preprocessing & feature extraction
	•	NumPy / SciPy / Pandas – numerical processing
	•	TensorFlow / Keras – CNN (ResNet50)
	•	Scikit-learn – SVM, scaling, evaluation

⸻

🔍 Feature Extraction Pipeline

The system extracts 14 handwriting features, mapped to 6 personality traits, using established graphology rules.

✏️ Extracted Handwriting Features
	1.	Slant Angle → Emotional Expression
	2.	Left Margin → Social Attachment
	3.	Top Margin → Ambition vs Calmness
	4.	Baseline Angle → Emotional Outlook
	5.	Letter Size → Introversion / Extroversion
	6.	Word Spacing → Social Interaction
	7.	Line Spacing → Emotional Boundaries

Each feature is measured quantitatively using computer vision techniques such as:
	•	Edge detection (Canny)
	•	Hough Line Transform
	•	Contour analysis
	•	Projection profiles
	•	Linear regression

⸻

🧠 Personality Traits Predicted
	•	Emotional Scale (Suppressed / Balanced / Expressive / Unstable)
	•	Orientation (Goal-Oriented / Stable / Reserved / Independent)
	•	Emotional Outlook (Optimistic / Balanced / Discouraged)
	•	Letter Size Trait (Introvert / Extrovert)
	•	Social Isolation (Sociable / Balanced / Reserved / Clingy)
	•	Concentration (Focused / Relaxed / Distracted / Tense)

⸻

🤖 CNN Model Details
	•	Base Architecture: ResNet50 (ImageNet pretrained)
	•	Transfer Learning Strategy:
	•	Phase 1: Train custom top layers
	•	Phase 2: Fine-tune last 50 layers
	•	Embedding Size: 256-dimensional feature vector
	•	Data Augmentation: Rotation, zoom, shift, shear, brightness

The trained CNN is used only as a feature extractor, not a final classifier.

⸻

🔗 Hybrid Feature Vector

X_combined = concatenate(
    [cnn_features (256-D), handcrafted_features (14-D)],
    axis=1
)

This hybrid approach provides:
	•	CNN → automatic visual pattern learning
	•	Graphology features → domain knowledge & interpretability

⸻

📂 Project Structure

├── train.py                     # CNN training
├── trainsvm.py                  # Hybrid CNN + SVM training
├── testsvm.py                   # Inference
├── feature_extaraction.py       # Handcrafted feature extraction
├── evaluate_model.py            # Evaluation scripts
├── CNN_MODEL_EXPLANATION.md     # CNN design details
├── FEATURE_EXTRACTION_EXPLANATION.md
├── new0margin.py                # Margin detection
├── visual.py                    # Visualization utilities
└── README.md


⸻

📊 Advantages of This Approach
	•	✅ Combines deep learning + human knowledge
	•	✅ Interpretable personality predictions
	•	✅ Robust to small datasets via transfer learning
	•	✅ Suitable for academic research & real-world applications

⸻

📌 Applications
	•	Psychological assessment tools
	•	Behavioral analysis systems
	•	Educational & recruitment screening (research use only)
	•	Human–computer interaction studies

⸻

⚠️ Disclaimer

This project is intended for academic and research purposes only.
Personality traits inferred from handwriting should not be treated as clinical or psychological diagnoses.

⸻

📄 License

This project is released under the MIT License.


