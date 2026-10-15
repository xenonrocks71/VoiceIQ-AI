"""
Model loading and parallel inference for Voice IQ AI.
"""

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import joblib
import numpy as np

# Allow imports from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.utils import (  # noqa: E402
    FEATURE_DIM,
    SAMPLE_RATE,
    extract_features_from_array,
)

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

GENDER_MODEL_PATH = os.path.join(MODELS_DIR, "gender_model.joblib")
AGE_MODEL_PATH = os.path.join(MODELS_DIR, "age_model.joblib")
DEEPFAKE_MODEL_PATH = os.path.join(MODELS_DIR, "deepfake_model.joblib")

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


class VoicePredictor:
    """Loads three Random Forest classifiers and runs parallel inference."""

    def __init__(self, models_dir: str = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self.gender_model: Optional[Any] = None
        self.age_model: Optional[Any] = None
        self.deepfake_model: Optional[Any] = None
        self._loaded = False

    def load_models(self) -> bool:
        """Load all three joblib models. Returns True if all are available."""
        paths = {
            "gender": os.path.join(self.models_dir, "gender_model.joblib"),
            "age": os.path.join(self.models_dir, "age_model.joblib"),
            "deepfake": os.path.join(self.models_dir, "deepfake_model.joblib"),
        }

        missing = [name for name, path in paths.items() if not os.path.isfile(path)]
        if missing:
            logger.warning(
                "Missing model files: %s — predictions will fail until models are trained.",
                missing,
            )
            return False

        try:
            self.gender_model = joblib.load(paths["gender"])
            self.age_model = joblib.load(paths["age"])
            self.deepfake_model = joblib.load(paths["deepfake"])
            self._loaded = True
            logger.info("All three models loaded successfully from %s", self.models_dir)
            return True
        except Exception as exc:
            logger.error("Failed to load models: %s", exc)
            self._loaded = False
            return False

    @property
    def models_loaded(self) -> bool:
        return self._loaded

    def _predict_single(
        self,
        model: Any,
        features: np.ndarray,
        label_map: Optional[dict] = None,
    ) -> tuple[Any, float]:
        """
        Run a single model prediction and return (label, confidence).
        Confidence is the maximum class probability.
        """
        features_2d = features.reshape(1, -1)
        prediction = model.predict(features_2d)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_2d)[0]
            confidence = float(np.max(proba))
        else:
            confidence = 1.0

        if label_map:
            prediction = label_map.get(prediction, prediction)

        return prediction, confidence

    def _run_gender(self, features: np.ndarray) -> tuple[str, float]:
        raw, conf = self._predict_single(self.gender_model, features)
        label = str(raw).lower()
        if label not in ("male", "female"):
            label = "male" if "male" in label else "female"
        return label, conf

    def _run_age(self, features: np.ndarray) -> tuple[str, float]:
        raw, conf = self._predict_single(self.age_model, features)
        band = str(raw)
        if band not in AGE_BANDS:
            for b in AGE_BANDS:
                if b in band:
                    band = b
                    break
        return band, conf

    def _run_deepfake(self, features: np.ndarray) -> tuple[bool, str, float]:
        raw, conf = self._predict_single(self.deepfake_model, features)
        raw_str = str(raw).lower()

        is_real = raw_str in ("real", "1", "true", "human", "real human")
        if raw_str in ("fake", "0", "false", "ai", "ai generated", "deepfake"):
            is_real = False

        label = "Real Human" if is_real else "AI Generated"
        return is_real, label, conf

    def predict_from_array(self, y: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
        """
        Extract features from a numpy audio array and run all three classifiers
        in parallel.
        """
        if not self._loaded:
            raise RuntimeError(
                "Models are not loaded. Train models first and place them in models/."
            )

        logger.info("Extracting features from audio array (samples=%d, sr=%d)", len(y), sr)
        features = extract_features_from_array(y, sr=sr)

        if features.shape[0] != FEATURE_DIM:
            raise ValueError(
                f"Expected {FEATURE_DIM} features, got {features.shape[0]}"
            )

        logger.info("Running parallel inference on 3 classifiers")

        with ThreadPoolExecutor(max_workers=3) as executor:
            gender_future = executor.submit(self._run_gender, features)
            age_future = executor.submit(self._run_age, features)
            deepfake_future = executor.submit(self._run_deepfake, features)

            gender, gender_conf = gender_future.result()
            age_band, age_conf = age_future.result()
            is_real, deepfake_label, deepfake_conf = deepfake_future.result()

        pitch_hz = round(float(features[163]), 1)
        f1_hz = round(float(features[165]), 1)
        f2_hz = round(float(features[167]), 1)
        hnr_db = round(float(features[170]), 2)

        result = {
            "gender": gender,
            "age_band": age_band,
            "is_real": is_real,
            "label": deepfake_label,
            "confidence": {
                "gender": round(gender_conf, 4),
                "age": round(age_conf, 4),
                "deepfake": round(deepfake_conf, 4),
            },
            "metrics": {
                "pitch_hz": pitch_hz,
                "f1_formant_hz": f1_hz,
                "f2_formant_hz": f2_hz,
                "hnr_db": hnr_db,
            },
        }

        logger.info("Prediction result: %s", result)
        return result

    def predict_from_file(self, file_path: str) -> dict:
        """Load audio from disk, extract features, and predict."""
        from training.utils import load_audio

        y = load_audio(file_path, sr=SAMPLE_RATE)
        return self.predict_from_array(y, sr=SAMPLE_RATE)


# Module-level singleton used by the Flask app
_predictor: Optional[VoicePredictor] = None


def get_predictor() -> VoicePredictor:
    global _predictor
    if _predictor is None:
        _predictor = VoicePredictor()
        _predictor.load_models()
    return _predictor
