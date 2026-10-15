"""
VoiceIQ AI — Production Gunicorn Configuration
Automatically discovered and loaded by Gunicorn on startup.
"""
import os

# Network binding
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Worker topology: 1 worker process with 4 threads
# Strictly 1 worker to guarantee RAM usage stays around ~240MB (well below Render's 512MB ceiling)
workers = 1
threads = 4
worker_class = "gthread"

# Timeout: 180 seconds to comfortably avoid 502/504 Bad Gateway timeouts on CPU-throttled instances
timeout = 180
graceful_timeout = 30
keepalive = 5

# Logging: Stream to standard output and standard error for Render dashboard
loglevel = "info"
accesslog = "-"
errorlog = "-"
capture_output = True
enable_stdio_inheritance = True
