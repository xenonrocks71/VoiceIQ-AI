"""
VoiceIQ AI — Production WSGI Entrypoint.
Enables running directly or via Gunicorn:
  gunicorn app:app
  gunicorn api.app:app
"""
import os
import sys

# Ensure project root is at the head of sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.app import app  # noqa: E402, F401

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
