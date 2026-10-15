"""
Shared audio feature extraction for Voice IQ AI.

This module is used by both training scripts and the Flask API.
All audio is normalized to 16 kHz and clipped/padded to 5 seconds
before feature extraction. The output is a fixed 179-dimensional vector.
"""

import logging
import os
import warnings
from typing import Optional, Union

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
TARGET_DURATION_SEC = 5.0
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_DURATION_SEC)
FEATURE_DIM = 179

N_MFCC = 40
N_MEL_BANDS = 20
N_FFT = 2048
HOP_LENGTH = 512
LPC_ORDER = 12


# ---------------------------------------------------------------------------
# Audio loading & preprocessing
# ---------------------------------------------------------------------------
def load_audio(file_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file and resample to the target sample rate."""
    y, native_sr = librosa.load(file_path, sr=sr, mono=True)
    logger.debug(
        "Loaded audio from %s — native_sr=%s, target_sr=%s, samples=%d",
        file_path,
        native_sr,
        sr,
        len(y),
    )
    return y


def load_audio_from_bytes(
    audio_bytes: bytes,
    file_extension: str = ".wav",
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Load audio from raw bytes.  Used by the API when the client uploads a file.
    Writes to a temporary file because librosa needs a file path for some formats.
    """
    import tempfile

    ext = file_extension if file_extension.startswith(".") else f".{file_extension}"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        return load_audio(tmp_path, sr=sr)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def normalize_audio(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Clip or pad audio to exactly TARGET_DURATION_SEC seconds.
    Shorter clips are zero-padded; longer clips are centre-trimmed.
    """
    target_len = int(sr * TARGET_DURATION_SEC)

    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    elif len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start : start + target_len]

    # Peak-normalize to prevent clipping artefacts in feature computation
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Individual feature helpers
# ---------------------------------------------------------------------------
def _safe_mean(feature_matrix: np.ndarray, axis: int = 1) -> np.ndarray:
    """Compute per-dimension mean, returning zeros for empty matrices."""
    if feature_matrix.size == 0:
        return np.zeros(feature_matrix.shape[0] if feature_matrix.ndim > 1 else 1)
    return np.mean(feature_matrix, axis=axis)


def _safe_std(feature_matrix: np.ndarray, axis: int = 1) -> np.ndarray:
    """Compute per-dimension std, returning zeros for empty matrices."""
    if feature_matrix.size == 0:
        return np.zeros(feature_matrix.shape[0] if feature_matrix.ndim > 1 else 1)
    return np.std(feature_matrix, axis=axis)


def _extract_pitch(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Extract fundamental frequency mean and std using librosa.pyin."""
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

        if len(voiced_f0) == 0:
            return 0.0, 0.0

        return float(np.mean(voiced_f0)), float(np.std(voiced_f0))
    except Exception as exc:
        logger.warning("Pitch extraction failed: %s", exc)
        return 0.0, 0.0


def _lpc_formants(frame: np.ndarray, sr: int, order: int = LPC_ORDER) -> tuple[float, float]:
    """Estimate F1 and F2 for a single frame via LPC root analysis."""
    try:
        frame = frame * np.hamming(len(frame))
        a = librosa.lpc(frame, order=order)
        if a is None or len(a) < 2:
            return 0.0, 0.0

        roots = np.roots(a)
        roots = roots[np.imag(roots) >= 0]

        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = sorted(angles * (sr / (2 * np.pi)))
        freqs = [f for f in freqs if 90 < f < 5000]

        f1 = freqs[0] if len(freqs) > 0 else 0.0
        f2 = freqs[1] if len(freqs) > 1 else 0.0
        return float(f1), float(f2)
    except Exception:
        return 0.0, 0.0


def _extract_formants(y: np.ndarray, sr: int) -> tuple[float, float, float, float]:
    """
    Extract F1 and F2 mean + std across voiced frames.
    Returns (f1_mean, f1_std, f2_mean, f2_std).
    """
    frame_length = int(0.025 * sr)  # 25 ms
    hop = int(0.010 * sr)             # 10 ms

    f1_list, f2_list = [], []
    for start in range(0, len(y) - frame_length, hop):
        frame = y[start : start + frame_length]
        if np.max(np.abs(frame)) < 0.01:
            continue
        f1, f2 = _lpc_formants(frame, sr)
        if f1 > 0:
            f1_list.append(f1)
        if f2 > 0:
            f2_list.append(f2)

    f1_mean = float(np.mean(f1_list)) if f1_list else 0.0
    f1_std = float(np.std(f1_list)) if f1_list else 0.0
    f2_mean = float(np.mean(f2_list)) if f2_list else 0.0
    f2_std = float(np.std(f2_list)) if f2_list else 0.0
    return f1_mean, f1_std, f2_mean, f2_std


def _extract_hnr(y: np.ndarray, sr: int) -> float:
    """
    Estimate Harmonics-to-Noise Ratio (HNR) in dB using the
    autocorrelation method on a single 5-second frame.
    """
    try:
        frame_length = min(len(y), int(0.04 * sr))
        if frame_length < 2:
            return 0.0

        # Use the middle frame for stability
        mid = len(y) // 2
        frame = y[mid : mid + frame_length]
        frame = frame * np.hamming(len(frame))

        r = np.correlate(frame, frame, mode="full")
        r = r[len(r) // 2 :]

        if r[0] == 0:
            return 0.0

        # Find first peak after the zeroth lag
        min_lag = int(sr / 500)   # 500 Hz upper bound for f0
        max_lag = int(sr / 50)    # 50 Hz lower bound for f0
        search = r[min_lag : max_lag + 1]

        if len(search) == 0:
            return 0.0

        peak = np.max(search)
        hnr = 10.0 * np.log10(peak / (r[0] - peak + 1e-10))
        return float(np.clip(hnr, -20.0, 40.0))
    except Exception as exc:
        logger.warning("HNR extraction failed: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------------------------
def extract_features_from_array(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Extract a 179-dimensional feature vector from a 1-D audio array.

    Feature breakdown (179 total):
      MFCC means              : 40
      MFCC delta means        : 40
      MFCC delta-delta means  : 40
      Chroma STFT means       : 12
      Mel spectrogram means   : 20  (first 20 bands)
      Spectral contrast means :  7
      Zero crossing rate      :  1  (mean)
      Spectral rolloff          :  1  (mean)
      Spectral bandwidth        :  1  (mean)
      Spectral centroid         :  1  (mean)
      Pitch (f0 mean + std)     :  2
      Formants F1/F2 (mean+std) :  4
      RMS energy                :  1  (mean)
      Harmonics-to-noise ratio  :  1
      Tonnetz means             :  6
      ZCR std + RMS std         :  2
    """
    y = normalize_audio(y, sr)

    features: list[float] = []

    # --- MFCCs (40 + 40 + 40 = 120) ---
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    features.extend(_safe_mean(mfcc).tolist())        # 40
    features.extend(_safe_mean(mfcc_delta).tolist())  # 40
    features.extend(_safe_mean(mfcc_delta2).tolist()) # 40

    # --- Chroma STFT (12) ---
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    features.extend(_safe_mean(chroma).tolist())

    # --- Mel spectrogram — first 20 bands (20) ---
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    features.extend(_safe_mean(mel[:N_MEL_BANDS]).tolist())

    # --- Spectral contrast (7) ---
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    features.extend(_safe_mean(contrast).tolist())

    # --- Zero crossing rate (1 mean) ---
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    features.append(float(_safe_mean(zcr)))

    # --- Spectral rolloff (1) ---
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    features.append(float(_safe_mean(rolloff)))

    # --- Spectral bandwidth (1) ---
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    features.append(float(_safe_mean(bandwidth)))

    # --- Spectral centroid (1) ---
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    features.append(float(_safe_mean(centroid)))

    # --- Pitch — f0 mean + std (2) ---
    f0_mean, f0_std = _extract_pitch(y, sr)
    features.extend([f0_mean, f0_std])

    # --- Formants F1 + F2 via LPC — mean + std each (4) ---
    f1_m, f1_s, f2_m, f2_s = _extract_formants(y, sr)
    features.extend([f1_m, f1_s, f2_m, f2_s])

    # --- RMS energy (1 mean) ---
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    features.append(float(_safe_mean(rms)))

    # --- Harmonics-to-noise ratio (1) ---
    features.append(_extract_hnr(y, sr))

    # --- Tonnetz (6) ---
    tonnetz = librosa.feature.tonnetz(
        y=librosa.effects.harmonic(y), sr=sr
    )
    features.extend(_safe_mean(tonnetz).tolist())

    # --- Supplementary temporal stats (2) → brings total to 179 ---
    features.append(float(_safe_std(zcr)))
    features.append(float(_safe_std(rms)))

    feature_vector = np.array(features, dtype=np.float32)

    if feature_vector.shape[0] != FEATURE_DIM:
        raise ValueError(
            f"Feature dimension mismatch: expected {FEATURE_DIM}, "
            f"got {feature_vector.shape[0]}"
        )

    # Replace NaN / Inf with 0
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_vector


def extract_features(file_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load an audio file and extract the 179-dimensional feature vector.

    Parameters
    ----------
    file_path : str
        Path to a WAV, MP3, OGG, FLAC, or other librosa-supported audio file.
    sr : int
        Target sample rate (default 16000 Hz).

    Returns
    -------
    np.ndarray
        Shape (179,) float32 feature vector.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    logger.info("Extracting features from: %s", file_path)
    y = load_audio(file_path, sr=sr)
    return extract_features_from_array(y, sr=sr)


def extract_features_batch(
    file_paths: list[str],
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Extract features for a list of audio files.
    Returns an (N, 179) ndarray.
    """
    vectors = []
    for path in file_paths:
        try:
            vectors.append(extract_features(path, sr=sr))
        except Exception as exc:
            logger.error("Failed to extract features from %s: %s", path, exc)
    return np.vstack(vectors) if vectors else np.empty((0, FEATURE_DIM), dtype=np.float32)


# ---------------------------------------------------------------------------
# Age-band mapping helper (used during training)
# ---------------------------------------------------------------------------
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

# Mozilla Common Voice age string → default band
CV_AGE_MAP: dict[str, str] = {
    "teens": "16-20",
    "twenties": "21-25",
    "thirties": "31-35",
    "fourties": "36-40",
    "fifties": "41-45",
    "sixties": "41-45",
    "seventies": "41-45",
    "eighties": "41-45",
    "nineties": "41-45",
}


def map_cv_age(age_label: str, duration_sec: float = 0.0) -> Optional[str]:
    """
    Map a Mozilla Common Voice age metadata value to one of the 8 bands.
    For 'twenties', uses audio duration heuristic to split 21-25 vs 26-30.
    """
    if not age_label or str(age_label).strip() == "":
        return None

    age_label = str(age_label).strip().lower()

    if age_label == "twenties":
        return "21-25" if duration_sec < 3.5 else "26-30"

    return CV_AGE_MAP.get(age_label)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: python utils.py <audio_file>")
        print(f"Extracts {FEATURE_DIM}-dim feature vector.")
        sys.exit(1)

    vec = extract_features(sys.argv[1])
    print(f"Feature vector shape: {vec.shape}")
    print(f"First 10 values: {vec[:10]}")
    print(f"Min: {vec.min():.4f}  Max: {vec.max():.4f}  Mean: {vec.mean():.4f}")
