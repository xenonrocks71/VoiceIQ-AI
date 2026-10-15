"""
Voice IQ AI — Flask REST API

Endpoints:
  GET  /health   — service health check
  POST /predict  — multipart audio upload → gender / age / deepfake predictions
"""

import io
import logging
import os
import sys
import tempfile
import traceback
import subprocess
from typing import Optional

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

# ---------------------------------------------------------------------------
# FFmpeg setup via imageio-ffmpeg
# ---------------------------------------------------------------------------
try:
    import imageio_ffmpeg
    import stat

    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
    if os.path.exists(FFMPEG_EXE):
        try:
            cur_mode = os.stat(FFMPEG_EXE).st_mode
            os.chmod(FFMPEG_EXE, cur_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except Exception:
            pass

    ffmpeg_dir = os.path.dirname(FFMPEG_EXE)
    if ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

    # Ensure both 'ffmpeg' and 'ffmpeg.exe' aliases exist in directory
    for alias_name in ("ffmpeg", "ffmpeg.exe"):
        alias_path = os.path.join(ffmpeg_dir, alias_name)
        if not os.path.exists(alias_path) and os.path.exists(FFMPEG_EXE):
            try:
                import shutil
                shutil.copyfile(FFMPEG_EXE, alias_path)
                os.chmod(alias_path, 0o755)
            except Exception:
                pass
except Exception as _ffmpeg_exc:
    FFMPEG_EXE = "ffmpeg"

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.predictor import get_predictor  # noqa: E402
from training.utils import SAMPLE_RATE, load_audio  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(PROJECT_ROOT, "api", "voice_iq_api.log"),
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("voice_iq_api")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app, resources={r"/*": {"origins": "*"}})

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".webm", ".flac", ".m4a", ".aac"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


@app.route("/", methods=["GET"])
def index():
    """Serve the single-page frontend application."""
    return app.send_static_file("index.html")


def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def convert_to_wav_16k(
    input_bytes: bytes,
    input_ext: str,
) -> str:
    """
    Convert uploaded audio bytes to a 16 kHz mono WAV temp file.
    Uses FFmpeg / pydub with bundled imageio-ffmpeg for robust multi-format conversion.
    Returns the path to the temp WAV file (caller must delete).
    """
    input_ext = input_ext.lower()
    tmp_in_path: Optional[str] = None
    tmp_out_path: Optional[str] = None

    try:
        # Fast-path for WAV: if soundfile can decode it in memory, write 16kHz WAV directly
        if input_ext in (".wav", ""):
            try:
                import soundfile as sf
                data, in_sr = sf.read(io.BytesIO(input_bytes))
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                if in_sr != SAMPLE_RATE:
                    data = librosa.resample(data, orig_sr=in_sr, target_sr=SAMPLE_RATE)
                return _write_wav(data.astype(np.float32), SAMPLE_RATE)
            except Exception as wav_err:
                logger.warning("In-memory WAV decode skipped: %s. Falling back to disk/FFmpeg.", wav_err)

        # Write incoming audio to a temp file
        with tempfile.NamedTemporaryFile(suffix=input_ext, delete=False) as tmp:
            tmp.write(input_bytes)
            tmp_in_path = tmp.name

        # Output WAV path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_out_path = tmp.name

        # Direct FFmpeg transcoding to 16 kHz mono 16-bit PCM WAV
        cmd = [
            FFMPEG_EXE,
            "-nostdin",
            "-y",
            "-i",
            tmp_in_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            tmp_out_path,
        ]
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )

        if proc.returncode != 0:
            err_msg = proc.stderr.decode("utf-8", errors="ignore") if proc.stderr else "FFmpeg failed"
            logger.warning("FFmpeg transcoding non-zero returncode %d: %s. Trying pydub...", proc.returncode, err_msg)
            from pydub import AudioSegment
            AudioSegment.converter = FFMPEG_EXE
            audio = AudioSegment.from_file(tmp_in_path)
            audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
            audio.export(tmp_out_path, format="wav")

        logger.info(
            "Converted %s -> 16 kHz mono WAV (%s)",
            input_ext,
            tmp_out_path,
        )
        return tmp_out_path

    except Exception as exc:
        for path in (tmp_in_path, tmp_out_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        raise RuntimeError(f"Audio conversion failed: {exc}") from exc
    finally:
        if tmp_in_path and os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)


def _write_wav(y, sr: int) -> str:
    """Write a numpy array to a temporary WAV file and return its path."""
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, y, sr)
        return tmp.name


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    predictor = get_predictor()
    status = {
        "status": "ok",
        "models_loaded": predictor.models_loaded,
    }
    logger.info("Health check: %s", status)
    return jsonify(status), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept multipart/form-data with an 'audio' file field.
    Returns gender, age_band, deepfake label, and confidence scores.
    """
    predictor = get_predictor()

    if not predictor.models_loaded:
        logger.error("Prediction requested but models are not loaded")
        return jsonify({
            "error": "Models not loaded. Please train models and place them in the models/ directory.",
            "models_loaded": False,
        }), 503

    if "audio" not in request.files:
        logger.warning("POST /predict — missing 'audio' field")
        return jsonify({"error": "No audio file provided. Send multipart form with 'audio' field."}), 400

    audio_file = request.files["audio"]

    if audio_file.filename == "" or audio_file.filename is None:
        return jsonify({"error": "Empty filename. Please upload a valid audio file."}), 400

    filename = audio_file.filename
    ext = os.path.splitext(filename)[1].lower()

    if not allowed_file(filename):
        return jsonify({
            "error": f"Unsupported file type '{ext}'. "
                     f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    wav_path: Optional[str] = None

    try:
        audio_bytes = audio_file.read()
        if len(audio_bytes) == 0:
            return jsonify({"error": "Uploaded file is empty."}), 400

        logger.info(
            "Received audio upload: filename=%s, size=%d bytes",
            filename,
            len(audio_bytes),
        )

        wav_path = convert_to_wav_16k(audio_bytes, ext)
        y = load_audio(wav_path, sr=SAMPLE_RATE)

        result = predictor.predict_from_array(y, sr=SAMPLE_RATE)
        return jsonify(result), 200

    except RuntimeError as exc:
        logger.error("Audio processing error: %s", exc)
        return jsonify({"error": str(exc)}), 422

    except Exception as exc:
        logger.error("Prediction failed: %s\n%s", exc, traceback.format_exc())
        return jsonify({"error": f"Internal server error: {exc}"}), 500

    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum upload size is 50 MB."}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(500)
def internal_error(e):
    logger.error("Unhandled 500 error: %s", e)
    return jsonify({"error": "Internal server error."}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Voice IQ AI on http://0.0.0.0:%d", port)
    get_predictor()  # Eager-load models at startup
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
