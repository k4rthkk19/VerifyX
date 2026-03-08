"""
models/detector.py
==================
Core prediction engine.

Strategy
--------
1. Try to load a fine-tuned DistilBERT model from HuggingFace Transformers.
2. If that fails (no GPU / memory / network), fall back to a classic ML pipeline
   built with scikit-learn + TF-IDF.  The fallback is trained on a small
   synthetic dataset just to keep the demo functional — replace with a real
   labelled corpus for production use.

Both paths expose the same interface:
    result = detector.predict(text)
    # → {"prediction": "FAKE"|"REAL",
    #    "confidence": 0.0–100.0,
    #    "explanation": "...",
    #    "model_used": "..."}
"""
import torch
import re
import logging
import os

logger = logging.getLogger(__name__)

# ─── Linguistic Feature Helpers ───────────────────────────────────────────────

SENSATIONAL_WORDS = {
    "shocking", "unbelievable", "jaw-dropping", "bombshell", "explosive",
    "secret", "conspiracy", "hoax", "exposed", "cover-up", "scandal",
    "breaking", "urgent", "alert", "warning", "danger", "crisis",
    "miracle", "cure", "banned", "censored", "suppressed",
    "they don't want you to know", "wake up", "sheeple",
    "fake media", "mainstream media lies", "deep state",
    "100%", "never before", "you won't believe",
}

CREDIBILITY_SIGNALS = {
    "according to", "research shows", "study finds", "published in",
    "peer-reviewed", "scientists", "experts", "university", "professor",
    "data indicates", "statistics show", "evidence suggests",
    "official statement", "government", "spokesperson",
}

def analyse_linguistic_features(text: str) -> dict:
    """
    Heuristic analysis of text for red-flag patterns.
    Returns a dict of flags and a short explanation string.
    """
    lower = text.lower()
    words = re.findall(r"\b\w+\b", lower)
    word_set = set(words)

    # Count sensational / credible cues
    sensational_hits = [w for w in SENSATIONAL_WORDS if w in lower]
    credibility_hits = [w for w in CREDIBILITY_SIGNALS if w in lower]

    # Excessive capitalisation (SHOUTING)
    upper_words = [w for w in text.split() if w.isupper() and len(w) > 2]
    upper_ratio  = len(upper_words) / max(len(text.split()), 1)

    # Excessive exclamation / question marks
    exclaim_count = text.count("!")
    question_count = text.count("?")

    flags = []
    if sensational_hits:
        flags.append(f"Sensational language detected: {', '.join(sensational_hits[:3])}")
    if upper_ratio > 0.15:
        flags.append("Excessive capitalisation (common in misleading content)")
    if exclaim_count > 3:
        flags.append(f"Multiple exclamation marks ({exclaim_count}) detected")
    if question_count > 4:
        flags.append(f"Multiple rhetorical questions ({question_count}) detected")
    if credibility_hits:
        flags.append(f"Credibility signals present: {', '.join(credibility_hits[:3])}")

    score = (
        len(sensational_hits) * 2
        + (upper_ratio > 0.15) * 3
        + min(exclaim_count, 5)
        + min(question_count // 2, 3)
        - len(credibility_hits) * 2
    )

    return {
        "sensational_hits": sensational_hits,
        "credibility_hits": credibility_hits,
        "upper_ratio":      round(upper_ratio, 3),
        "exclaim_count":    exclaim_count,
        "flags":            flags,
        "linguistic_score": score,
    }

# ─── DistilBERT Classifier ────────────────────────────────────────────────────

class DistilBERTClassifier:
    """
    Wraps HuggingFace pipeline for zero-shot / fine-tuned text classification.

    For a real deployment you would fine-tune on the LIAR or FakeNewsNet
    dataset.  Here we use the zero-shot pipeline with two candidate labels
    to stay dependency-light.
    """

    def __init__(self):
        from transformers import pipeline
        logger.info("Loading DistilBERT zero-shot pipeline …")
        self.pipe = pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
            device=-1,          # CPU; change to 0 for GPU
        )
        self.pipe("test", candidate_labels=["real news", "fake news"])

        self.labels = ["real news", "fake news"]
        logger.info("DistilBERT pipeline loaded.")

    def predict(self, text: str) -> dict:
        # Truncate to avoid OOM on very long texts
        truncated = text[:300]

        with torch.no_grad():
            out = self.pipe(truncated, candidate_labels=self.labels)
        top_label = out["labels"][0]
        top_score = out["scores"][0]

        prediction = "FAKE" if "fake" in top_label else "REAL"
        confidence = round(top_score * 100, 2)

        features = analyse_linguistic_features(truncated)
        explanation = self._build_explanation(prediction, confidence, features)

        return {
            "prediction":  prediction,
            "confidence":  confidence,
            "explanation": explanation,
            "model_used":  "DistilBERT (zero-shot MNLI)",
            "features":    features,
        }

    def _build_explanation(self, prediction, confidence, features):
        parts = []
        if prediction == "FAKE":
            parts.append(
                f"The model classifies this content as likely fabricated "
                f"with {confidence:.1f}% confidence."
            )
        else:
            parts.append(
                f"The model classifies this content as likely credible "
                f"with {confidence:.1f}% confidence."
            )
        parts.extend(features["flags"])
        if not features["flags"]:
            parts.append("No strong linguistic red flags detected.")
        return " | ".join(parts)

# ─── Scikit-learn Fallback ────────────────────────────────────────────────────

_FAKE_SAMPLES = [
    "SHOCKING: Scientists BANNED from revealing the truth about vaccines!!",
    "You won't BELIEVE what they found in the water supply — cover-up exposed!",
    "Breaking: Deep state conspiracy to silence all citizens REVEALED",
    "Miracle cure that doctors don't want you to know about finally exposed",
    "WAKE UP SHEEPLE: The moon landing was completely staged!!",
    "Secret government document proves aliens are living among us",
    "They are putting chemicals in the water to control your mind!!",
    "Bombshell: mainstream media LIES about everything, here's the truth",
    "This banned video will change everything you know about cancer cures",
    "Unbelievable scandal: politician caught in massive cover-up operation",
    "Warning: 5G towers are actually mind control devices, experts confirm",
    "URGENT: The cure for all diseases suppressed by Big Pharma for decades",
]

_REAL_SAMPLES = [
    "According to a peer-reviewed study published in Nature, researchers found evidence of climate change.",
    "The Federal Reserve announced a quarter-point interest rate increase on Wednesday.",
    "University scientists released data indicating a potential breakthrough in Alzheimer's research.",
    "Official government statistics show unemployment fell to 3.5% last month.",
    "A spokesperson for the company confirmed the merger in an official statement.",
    "Experts at the WHO published findings suggesting the new treatment is effective.",
    "Research shows that regular exercise reduces the risk of cardiovascular disease.",
    "The court issued a ruling following weeks of testimony from both sides.",
    "Scientists at CERN reported successful particle collision experiments.",
    "According to census data, the population grew by 2% over the past decade.",
    "The prime minister addressed parliament regarding the new economic policy.",
    "Investigators published their findings after a two-year independent inquiry.",
]

class SklearnClassifier:
    """
    TF-IDF + Logistic Regression fallback classifier.
    Trained on a minimal synthetic corpus — upgrade to a real dataset!
    """

    def __init__(self):
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        import numpy as np

        texts  = _FAKE_SAMPLES + _REAL_SAMPLES
        labels = (["FAKE"] * len(_FAKE_SAMPLES)) + (["REAL"] * len(_REAL_SAMPLES))

        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ])
        self.model.fit(texts, labels)
        logger.info("Sklearn fallback classifier trained on %d samples.", len(texts))

    def predict(self, text: str) -> dict:
        proba      = self.model.predict_proba([text])[0]
        classes    = self.model.classes_          # e.g. ['FAKE', 'REAL']
        pred_idx   = proba.argmax()
        prediction = classes[pred_idx]
        confidence = round(float(proba[pred_idx]) * 100, 2)

        features    = analyse_linguistic_features(text)
        explanation = self._build_explanation(prediction, confidence, features)

        return {
            "prediction":  prediction,
            "confidence":  confidence,
            "explanation": explanation,
            "model_used":  "TF-IDF + Logistic Regression (fallback)",
            "features":    features,
        }

    def _build_explanation(self, prediction, confidence, features):
        parts = []
        if prediction == "FAKE":
            parts.append(
                f"Linguistic patterns suggest fabricated content "
                f"({confidence:.1f}% confidence)."
            )
        else:
            parts.append(
                f"Content appears consistent with factual reporting "
                f"({confidence:.1f}% confidence)."
            )
        parts.extend(features["flags"])
        if not features["flags"]:
            parts.append("Text reads with neutral, factual tone.")
        return " | ".join(parts)

# ─── Unified Detector ─────────────────────────────────────────────────────────

class FakeNewsDetector:
    """
    Tries DistilBERT first; falls back to sklearn if transformers are
    unavailable (no network, low RAM, missing package, etc.).
    """

    def __init__(self):
        self._backend = self._load_backend()

    def _load_backend(self):
        try:
            return DistilBERTClassifier()
        except Exception as exc:
            logger.warning(
                "DistilBERT unavailable (%s). Using sklearn fallback.", exc
            )
            return SklearnClassifier()

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            raise ValueError("Input text is empty.")
        return self._backend.predict(text.strip())
