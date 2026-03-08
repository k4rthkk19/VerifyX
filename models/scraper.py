"""
models/scraper.py
=================
Lightweight article scraper.
Uses requests + BeautifulSoup to extract readable text from a URL.
Falls back to raw paragraph text if newspaper3k is unavailable.
"""

import logging
import re
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

REQUEST_TIMEOUT = 8  # seconds


def scrape_article(url: str) -> str | None:
    """
    Attempt to retrieve and extract the main textual content of an article.
    Returns a string of text, or None on failure.
    """
    # ── Try newspaper3k first (best extraction) ─────────────────────────────
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        text = (article.title or "") + " " + (article.text or "")
        text = text.strip()
        if len(text) > 100:
            logger.info("newspaper3k extracted %d chars from %s", len(text), url)
            return text
    except Exception:
        pass  # fall through to BeautifulSoup

    # ── Fall back: raw requests + BeautifulSoup ──────────────────────────────
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Prefer <article> or <main>; fall back to all <p>
        container = soup.find("article") or soup.find("main") or soup
        paragraphs = container.find_all("p")
        text = " ".join(p.get_text(separator=" ") for p in paragraphs)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) > 100:
            logger.info("BeautifulSoup extracted %d chars from %s", len(text), url)
            return text

        logger.warning("Extracted text too short from %s", url)
        return None

    except Exception as exc:
        logger.error("Scraping failed for %s: %s", url, exc)
        return None
