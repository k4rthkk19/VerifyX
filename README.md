# 🔍 VerifyX — AI Fake News Detection App

A near-production prototype for **Final Year Project** demonstration.

---

## 📁 Project Structure

```
fakenews/
├── app.py                  # Flask app: routes, rate-limiting, DB writes
├── init_db.py              # One-time DB initialisation script
├── gunicorn.conf.py        # Production WSGI server config
├── requirements.txt        # Python dependencies
│
├── models/
│   ├── __init__.py
│   ├── detector.py         # FakeNewsDetector (DistilBERT + sklearn fallback)
│   └── scraper.py          # URL → article text extractor
│
├── templates/
│   ├── index.html          # Main app UI (paste text / URL analysis)
│   └── admin.html          # Admin dashboard (login + logs + charts)
│
├── static/                 # (optional) CSS/JS/img assets
│
└── database/
    └── fakenews.db         # SQLite database (auto-created)
```

---

## ⚡ Quick Start

### 1. Clone / extract the project
```bash
cd fakenews
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note on PyTorch**: For CPU-only, `torch` installs automatically.  
> For GPU support visit https://pytorch.org/get-started/locally/

### 4. Initialise the database
```bash
python init_db.py
```

### 5. Run the development server
```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🚀 Production Deployment

```bash
gunicorn -c gunicorn.conf.py app:app
```

### HTTPS (Nginx reverse proxy)
```nginx
server {
    listen 443 ssl;
    server_name yourdomain.com;

    ssl_certificate     /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;

    location / {
        proxy_pass         http://127.0.0.1:5000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
    }
}
```

---

## 🔐 Admin Dashboard

1. Navigate to **http://localhost:5000/admin**
2. Default password: `admin123`  
   ⚠️ **Change this before deployment!**  
   In `app.py`, update:
   ```python
   ADMIN_PASSWORD_HASH = hashlib.sha256(b"YOUR_NEW_PASSWORD").hexdigest()
   ```
3. The dashboard shows:
   - Total / Fake / Real counts, today's queries, avg confidence
   - Doughnut chart (Fake vs Real) + Bar chart (7-day trend)
   - Paginated query log table

---

## 🧠 AI Pipeline

```
User Input
    │
    ▼
Input Validation + XSS Sanitisation
    │
    ▼
(If URL) → Scrape article text (newspaper3k → BeautifulSoup fallback)
    │
    ▼
FakeNewsDetector.predict(text)
    ├── Try: DistilBERT zero-shot MNLI pipeline (HuggingFace Transformers)
    └── Fallback: TF-IDF + Logistic Regression (scikit-learn)
    │
    ▼
Linguistic Feature Analysis
  · Sensational word detection
  · Credibility signal detection
  · CAPITALISATION ratio
  · Exclamation / rhetorical question count
    │
    ▼
JSON Response → Frontend (Chart.js visualisation)
    │
    ▼
SQLite Logging
```

---

## 🛡️ Security Features

| Feature | Implementation |
|---|---|
| XSS Prevention | `html.escape()` + tag stripping on all inputs |
| SQL Injection | Parameterised queries (no string concatenation) |
| Rate Limiting | Sliding-window in-memory limiter (30 req/60s per IP) |
| Script Injection | Input length cap (10,000 chars) + regex validation |
| HTTPS Ready | Gunicorn + Nginx config provided |
| Admin Auth | SHA-256 hashed password + session tokens |

---

## 📊 API Reference

### `POST /api/predict`
```json
// Request
{ "text": "Breaking: Scientists BANNED from revealing the truth!!", "type": "text" }

// Response
{
  "prediction":  "FAKE",
  "confidence":  87.4,
  "explanation": "Linguistic patterns suggest fabricated content (87.4% confidence). Sensational language detected: shocking, banned",
  "model_used":  "DistilBERT (zero-shot MNLI)",
  "features": {
    "sensational_hits": ["shocking", "banned"],
    "credibility_hits": [],
    "flags":            ["Sensational language detected: shocking, banned"]
  }
}
```

### `GET /api/health`
```json
{ "status": "ok", "timestamp": "2024-01-01T12:00:00" }
```

### `POST /api/admin/login`
```json
// Request
{ "password": "admin123" }
// Response
{ "token": "abc123..." }
```

### `GET /api/admin/stats` *(requires X-Admin-Token header)*
### `GET /api/admin/logs?page=1&per_page=20` *(requires X-Admin-Token header)*

---

## 🔧 Customisation

### Use a fine-tuned model
Replace in `models/detector.py` → `DistilBERTClassifier.__init__()`:
```python
self.pipe = pipeline(
    "text-classification",
    model="path/to/your-finetuned-distilbert",
)
```
Train on [LIAR dataset](https://huggingface.co/datasets/liar) or [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) for best results.

### Change rate limits
In `app.py`:
```python
RATE_LIMIT_REQUESTS = 30   # max requests
RATE_LIMIT_WINDOW   = 60   # per N seconds
```

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| Flask | Web framework |
| transformers | HuggingFace DistilBERT |
| scikit-learn | Fallback ML model |
| newspaper3k | Article scraping |
| beautifulsoup4 | HTML parsing |
| Chart.js (CDN) | Frontend visualisation |
| SQLite3 | Query logging (stdlib) |

---

## ⚠️ Disclaimer

AI predictions are probabilistic — they are never 100% accurate. Always cross-reference with trusted sources. This prototype is designed for academic demonstration purposes.

---

*Built with Flask + DistilBERT · Final Year Project*
