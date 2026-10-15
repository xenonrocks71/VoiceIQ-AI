"""
Voice IQ AI — Baseline Model Generator
Generates initial trained scikit-learn RandomForest models (179-dim input)
and saves them to models/ so the API and Frontend work immediately out-of-the-box.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_DIM = 179
N_SAMPLES = 400

AGE_BANDS = [
    "5-10",
    "11-15",
    "16-20",
    "21-25",
    "26-30",
    "31-35",
    "36-40",
    "41-45",
]

def generate_models():
    print(f"Generating baseline models in {MODELS_DIR}...")
    np.random.seed(42)

    # -------------------------------------------------------------
    # 1. Gender Model (Male vs Female)
    # Feature 126 is pitch f0 mean (Males typically ~120 Hz, Females ~210 Hz)
    # -------------------------------------------------------------
    X_gender = np.random.randn(N_SAMPLES, FEATURE_DIM).astype(np.float32)
    # Synthetic pitch signal (f0 mean is at index 126)
    X_gender[: N_SAMPLES // 2, 126] = np.random.normal(120, 20, N_SAMPLES // 2)
    X_gender[N_SAMPLES // 2 :, 126] = np.random.normal(210, 25, N_SAMPLES // 2)

    y_gender = np.array(["male"] * (N_SAMPLES // 2) + ["female"] * (N_SAMPLES // 2))

    gender_clf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
    gender_clf.fit(X_gender, y_gender)
    gender_path = os.path.join(MODELS_DIR, "gender_model.joblib")
    joblib.dump(gender_clf, gender_path)
    print(f"  [OK] Gender model saved to: {gender_path}")

    # -------------------------------------------------------------
    # 2. Age Model (8 Age Bands)
    # -------------------------------------------------------------
    X_age = np.random.randn(N_SAMPLES, FEATURE_DIM).astype(np.float32)
    samples_per_band = N_SAMPLES // len(AGE_BANDS)
    y_age = []
    for i, band in enumerate(AGE_BANDS):
        y_age.extend([band] * samples_per_band)
        # Formants F1/F2 and pitch variation per age band
        X_age[i * samples_per_band : (i + 1) * samples_per_band, 126] = 250 - (i * 15)

    age_clf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
    age_clf.fit(X_age, y_age)
    age_path = os.path.join(MODELS_DIR, "age_model.joblib")
    joblib.dump(age_clf, age_path)
    print(f"  [OK] Age model saved to: {age_path}")

    # -------------------------------------------------------------
    # 3. Deepfake / Authenticity Model (Real Human vs AI Generated)
    # HNR is at index 132 (AI often has distinct noise/harmonic boundaries)
    # -------------------------------------------------------------
    X_fake = np.random.randn(N_SAMPLES, FEATURE_DIM).astype(np.float32)
    # Real speech has natural HNR and formant variations
    X_fake[: N_SAMPLES // 2, 132] = np.random.normal(18.0, 3.0, N_SAMPLES // 2)
    X_fake[N_SAMPLES // 2 :, 132] = np.random.normal(28.0, 4.0, N_SAMPLES // 2)

    y_fake = np.array(["Real Human"] * (N_SAMPLES // 2) + ["AI Generated"] * (N_SAMPLES // 2))

    fake_clf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
    fake_clf.fit(X_fake, y_fake)
    fake_path = os.path.join(MODELS_DIR, "deepfake_model.joblib")
    joblib.dump(fake_clf, fake_path)
    print(f"  [OK] Deepfake model saved to: {fake_path}")

    print("\nAll baseline models successfully generated and ready for inference!")

if __name__ == "__main__":
    generate_models()
