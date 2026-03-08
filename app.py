"""
Fake News Detection Web Application
=====================================
Main Flask application entry point.
Handles routing, API endpoints, rate limiting, and database logging.
"""

import os
import re
import time
import logging
import sqlite3
import hashlib
import html
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict
import threading

from flask import (
    Flask, request, jsonify, render_template,
    g, abort, session
)
from flask_cors import CORS

# ─── App Initialization ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

app.secret_key = os.environ.get("SECRET_KEY", os.urandom(32))

# Enable CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

DATABASE = os.path.join(os.path.dirname(__file__), "database", "fakenews.db")
RATE_LIMIT_REQUESTS = 30          # max requests
RATE_LIMIT_WINDOW   = 60          # per N seconds
ADMIN_PASSWORD_HASH = hashlib.sha256(b"admin123").hexdigest()  # Change in production

# ─── In-memory Rate Limiter ───────────────────────────────────────────────────

rate_store   = defaultdict(list)
rate_lock    = threading.Lock()

def is_rate_limited(ip: str) -> bool:
    """Sliding-window rate limiter — returns True if the IP is over limit."""
    now = time.time()
    with rate_lock:
        timestamps = rate_store[ip]
        # Drop timestamps outside window
        rate_store[ip] = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
        if len(rate_store[ip]) >= RATE_LIMIT_REQUESTS:
            return True
        rate_store[ip].append(now)
        return False

def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        ip = request.remote_addr or "unknown"
        if is_rate_limited(ip):
            return jsonify({
                "error": "Rate limit exceeded. Please wait before making more requests.",
                "retry_after": RATE_LIMIT_WINDOW
            }), 429
        return f(*args, **kwargs)
    return decorated

# ─── Database Helpers ─────────────────────────────────────────────────────────

def get_db():
    """Return a per-request SQLite connection (stored on Flask's g object)."""
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):
    db = g.pop("db", None)
    if db:
        db.close()

def init_db():
    """Create tables if they don't exist yet."""
    os.makedirs(os.path.dirname(DATABASE), exist_ok=True)
    with sqlite3.connect(DATABASE) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS queries (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ip_address  TEXT    NOT NULL,
                input_type  TEXT    NOT NULL,   -- 'text' or 'url'
                input_text  TEXT    NOT NULL,
                prediction  TEXT    NOT NULL,   -- 'FAKE' or 'REAL'
                confidence  REAL    NOT NULL,
                explanation TEXT,
                model_used  TEXT,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS admin_sessions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                token      TEXT UNIQUE NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
    logger.info("Database initialised at %s", DATABASE)

# ─── Input Sanitisation ───────────────────────────────────────────────────────

def sanitize_text(text: str) -> str:
    """Strip HTML tags and normalise whitespace to prevent XSS / injection."""
    text = html.escape(text)                        # encode special HTML chars
    text = re.sub(r"<[^>]+>", "", text)             # remove any residual tags
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_valid_url(url: str) -> bool:
    pattern = re.compile(
        r"^(https?://)?"                             # optional scheme
        r"(([a-zA-Z0-9\-]+\.)+[a-zA-Z]{2,})"       # domain
        r"(/[^\s]*)?$"                               # optional path
    )
    return bool(pattern.match(url))

# ─── Model Loading (lazy, thread-safe) ───────────────────────────────────────

_model_lock     = threading.Lock()
_classifier     = None   # holds the loaded detector instance

def get_classifier():
    """Return a cached classifier; initialise on first call."""
    global _classifier
    if _classifier is None:
        with _model_lock:
            if _classifier is None:          # double-checked locking
                from models.detector import FakeNewsDetector
                logger.info("Loading FakeNewsDetector …")
                _classifier = FakeNewsDetector()
                logger.info("FakeNewsDetector ready.")
    return _classifier

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/admin")
def admin():
    return render_template("admin.html")

# ── Health check ─────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})

# ── Main prediction endpoint ──────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
@rate_limit
def predict():
    """
    Accepts JSON body:
        { "text": "...",  "type": "text"|"url" }
    Returns:
        { "prediction": "FAKE"|"REAL", "confidence": float,
          "explanation": str, "model_used": str }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON payload."}), 400

    input_type = data.get("type", "text")
    raw_input  = data.get("text", "").strip()

    # ── Validation ──────────────────────────────────────────────────────────
    if not raw_input:
        return jsonify({"error": "Input text cannot be empty."}), 400
    if len(raw_input) > 10_000:
        return jsonify({"error": "Input exceeds maximum length (10 000 chars)."}), 400
    if input_type == "url" and not is_valid_url(raw_input):
        return jsonify({"error": "Invalid URL format."}), 400

    # ── Sanitise ────────────────────────────────────────────────────────────
    clean_input = sanitize_text(raw_input)

    # ── If URL, attempt to scrape headline / body text ─────────────────────
    analysis_text = clean_input
    if input_type == "url":
        try:
            from models.scraper import scrape_article
            analysis_text = scrape_article(raw_input) or clean_input
        except Exception as exc:
            logger.warning("Scraping failed for %s: %s", raw_input, exc)
            analysis_text = clean_input

    # ── Run classifier ──────────────────────────────────────────────────────
    try:
        clf    = get_classifier()
        result = clf.predict(analysis_text)
    except Exception as exc:
        logger.error("Prediction error: %s", exc, exc_info=True)
        return jsonify({"error": "Model inference failed. Please try again."}), 500

    # ── Persist to DB ────────────────────────────────────────────────────────
    try:
        db = get_db()
        db.execute(
            """INSERT INTO queries
               (ip_address, input_type, input_text, prediction,
                confidence, explanation, model_used)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                request.remote_addr or "unknown",
                input_type,
                clean_input[:500],            # store truncated version
                result["prediction"],
                result["confidence"],
                result.get("explanation", ""),
                result.get("model_used", "unknown"),
            )
        )
        db.commit()
    except Exception as exc:
        logger.warning("DB write error: %s", exc)

    return jsonify(result)

# ── Admin: authentication ─────────────────────────────────────────────────────

@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    data     = request.get_json(silent=True) or {}
    password = data.get("password", "")
    if hashlib.sha256(password.encode()).hexdigest() == ADMIN_PASSWORD_HASH:
        token = hashlib.sha256(os.urandom(32)).hexdigest()
        db    = get_db()
        db.execute("INSERT INTO admin_sessions (token) VALUES (?)", (token,))
        db.commit()
        return jsonify({"token": token})
    return jsonify({"error": "Invalid password."}), 401

def require_admin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("X-Admin-Token", "")
        if not token:
            return jsonify({"error": "Unauthorised."}), 401
        db  = get_db()
        row = db.execute(
            "SELECT id FROM admin_sessions WHERE token=?", (token,)
        ).fetchone()
        if not row:
            return jsonify({"error": "Unauthorised."}), 401
        return f(*args, **kwargs)
    return decorated

# ── Admin: logs ───────────────────────────────────────────────────────────────

@app.route("/api/admin/logs")
@require_admin
def admin_logs():
    page     = max(1, int(request.args.get("page", 1)))
    per_page = min(100, int(request.args.get("per_page", 20)))
    offset   = (page - 1) * per_page

    db    = get_db()
    total = db.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
    rows  = db.execute(
        "SELECT * FROM queries ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (per_page, offset)
    ).fetchall()

    return jsonify({
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "logs":     [dict(r) for r in rows],
    })

@app.route("/api/admin/stats")
@require_admin
def admin_stats():
    db = get_db()
    total    = db.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
    fake_cnt = db.execute("SELECT COUNT(*) FROM queries WHERE prediction='FAKE'").fetchone()[0]
    real_cnt = db.execute("SELECT COUNT(*) FROM queries WHERE prediction='REAL'").fetchone()[0]
    today    = db.execute(
        "SELECT COUNT(*) FROM queries WHERE DATE(created_at)=DATE('now')"
    ).fetchone()[0]
    avg_conf = db.execute("SELECT AVG(confidence) FROM queries").fetchone()[0] or 0
    recent   = db.execute(
        """SELECT DATE(created_at) as day, COUNT(*) as cnt
           FROM queries
           GROUP BY day
           ORDER BY day DESC
           LIMIT 7"""
    ).fetchall()

    return jsonify({
        "total":       total,
        "fake":        fake_cnt,
        "real":        real_cnt,
        "today":       today,
        "avg_conf":    round(avg_conf, 2),
        "daily_trend": [dict(r) for r in recent],
    })

# ─── Bootstrap ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    # Warm up the model before serving traffic
    logger.info("Warming up model …")
    try:
        get_classifier()
    except Exception as exc:
        logger.error("Model warm-up failed: %s", exc)
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,          # set True for development
        threaded=True,        # handle concurrent requests
    )
