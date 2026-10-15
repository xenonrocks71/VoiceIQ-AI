"""
VoiceIQ AI — Production WSGI Entrypoint for Gunicorn/uWSGI:
  gunicorn wsgi:app
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.app import app  # noqa: E402, F401
