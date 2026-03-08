# gunicorn.conf.py
# ================
# Production server configuration for VerifyX.
# Run: gunicorn -c gunicorn.conf.py app:app

import multiprocessing

# --- Server socket ---
bind    = "0.0.0.0:5000"
backlog = 2048

# --- Workers ---
# (2 × CPU cores) + 1 is a standard starting point.
# Keep at 1 if the DistilBERT model is large and memory is tight.
workers         = min(4, multiprocessing.cpu_count() * 2 + 1)
worker_class    = "sync"          # use 'gevent' for higher concurrency
worker_connections = 1000
timeout         = 60              # seconds; increase if model is slow to load
keepalive       = 5

# --- Process naming ---
proc_name = "verifyx"

# --- Logging ---
accesslog = "-"   # stdout
errorlog  = "-"   # stdout
loglevel  = "info"

# --- Security ---
limit_request_line   = 4096
limit_request_fields = 100
