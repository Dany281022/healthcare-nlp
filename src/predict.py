# src/predict.py
"""
Inference module — 3 tâches du projet Cambrian Healthcare NLP.

Task 1 : predict_sentiment()   — classification binaire (Positive/Negative)
Task 2 : predict_theme()       — classification de thème (4 classes UCI)
Task 3 : get_topics_nmf()      — topics latents NMF depuis le dataset

Dashboard helpers :
  get_topic_distribution()     — stats sentiment par thème
  get_real_samples()           — vrais feedbacks pour le LLM
  generate_llm_insight()       — insight LLM depuis feedbacks réels
"""
import json
import functools
import joblib
import pandas as pd
from config import (
    MODEL_PATH,
    SENTIMENT_MODEL_PATH,
    THEME_MODEL_PATH,
    NMF_MODEL_PATH,
    TFIDF_PATH,
    DATA_PATH,
    THEMES,
    TEXT_COL,
    LABEL_COL,
    THEME_COL,
    SATISFACTION_COL,
)
from src.llm_client import call_llm


# ── Helpers ────────────────────────────────────────────
def _load_vectorizer():
    """Load shared TF-IDF vectorizer."""
    try:
        return joblib.load(TFIDF_PATH)
    except FileNotFoundError:
        return joblib.load(MODEL_PATH.replace(".pkl", "_vectorizer.pkl"))


def _load_dataset(data_path: str = DATA_PATH) -> pd.DataFrame:
    """Load dataset — auto-detects CSV or XLSX."""
    if data_path.endswith(".xlsx"):
        return pd.read_excel(data_path)
    return pd.read_csv(data_path)


# ══════════════════════════════════════════════════════
# TASK 1 — Sentiment Prediction
# ══════════════════════════════════════════════════════
def predict_sentiment(text: str, model, vectorizer) -> dict:
    """
    Task 1 : predict sentiment for a single feedback string.

    Args:
        text:       Raw patient feedback.
        model:      Trained sentiment classifier (injected from main.py).
        vectorizer: Fitted TF-IDF vectorizer (injected from main.py).
    Returns:
        dict — text, prediction (0/1), label (Positive/Negative)
    """
    from src.preprocess import clean_text
    cleaned = clean_text(text)
    X       = vectorizer.transform([cleaned])
    pred    = model.predict(X)[0]
    label   = "Positive" if pred == 1 else "Negative"
    return {"text": text, "prediction": int(pred), "label": label}


# ══════════════════════════════════════════════════════
# TASK 2 — Theme Prediction
# ══════════════════════════════════════════════════════
def predict_theme(text: str, vectorizer) -> dict:
    """
    Task 2 : predict healthcare theme for a single feedback string.
    Themes : Anxiety, Birth Control, Depression, Pain
    """
    from src.preprocess import clean_text
    cleaned     = clean_text(text)
    X           = vectorizer.transform([cleaned])
    theme_model = joblib.load(THEME_MODEL_PATH)
    pred        = theme_model.predict(X)[0]
    proba       = theme_model.predict_proba(X)[0]
    confidence  = round(float(max(proba)), 3)
    return {"text": text, "theme": str(pred), "confidence": confidence}


# ══════════════════════════════════════════════════════
# TASK 3 — NMF Topic Modeling
# ══════════════════════════════════════════════════════
def get_topics_nmf() -> dict:
    """
    Task 3 : return latent topics extracted by NMF.
    Loads pre-computed topics from JSON artifact saved during training.
    Falls back to loading the NMF model directly if JSON not found.
    """
    topics_path = MODEL_PATH.replace("sentiment_model.pkl", "nmf_topics.json")
    try:
        with open(topics_path) as f:
            return json.load(f)
    except FileNotFoundError:
        pass

    # Fallback : recompute from saved model
    vectorizer    = _load_vectorizer()
    nmf_model     = joblib.load(NMF_MODEL_PATH)
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for i, topic in enumerate(nmf_model.components_):
        top_words = [feature_names[j] for j in topic.argsort()[-10:][::-1]]
        topics[f"topic_{i+1}"] = top_words
    return topics


# ══════════════════════════════════════════════════════
# Dashboard helpers
# ══════════════════════════════════════════════════════
@functools.lru_cache(maxsize=1)
def get_topic_distribution(data_path: str = DATA_PATH) -> dict:
    """
    Return sentiment distribution grouped by Theme.
    Cached after first call — no repeated disk reads.

    Handles missing or all-NaN Satisfaction column gracefully.
    """
    try:
        df = _load_dataset(data_path)
    except Exception as e:
        print(f"[ERROR] Cannot load dataset from '{data_path}': {e}")
        return {}

    result = {}
    for theme in THEMES:
        subset = df[df[THEME_COL] == theme]
        pos    = int((subset[LABEL_COL] == 1).sum())
        neg    = int((subset[LABEL_COL] == 0).sum())

        # Robust avg_satisfaction — handles missing column or all-NaN values
        if (
            SATISFACTION_COL in df.columns
            and not subset[SATISFACTION_COL].isna().all()
        ):
            avg_sat = round(float(subset[SATISFACTION_COL].dropna().mean()), 2)
        else:
            # Derive from sentiment ratio if Satisfaction column is absent
            total   = pos + neg
            avg_sat = round((pos / total) * 5, 2) if total > 0 else 0.0

        result[theme] = {
            "total":            len(subset),
            "positive":         pos,
            "negative":         neg,
            "avg_satisfaction": avg_sat,
        }
    return result


def get_real_samples(
    theme: str,
    n: int = 6,
    data_path: str = DATA_PATH,
) -> list[str]:
    """
    Return n real feedback texts for a given theme.
    Used by /insights endpoint — feeds actual data to the LLM.
    """
    df     = _load_dataset(data_path)
    subset = df[df[THEME_COL] == theme][TEXT_COL].dropna()
    return subset.sample(min(n, len(subset)), random_state=42).tolist()


def generate_llm_insight(feedback_samples: list[str], theme: str = "general") -> str:
    """
    Generate an LLM-powered insight from a list of real feedback strings.
    """
    joined = "\n- ".join(feedback_samples[:10])
    prompt = f"""You are a healthcare quality analyst.
Analyze the following patient feedback samples related to: {theme}.

Feedback:
- {joined}

Provide a concise insight (3-5 sentences) covering:
1. Main recurring issues or praise
2. Actionable recommendation for the healthcare team
3. Overall sentiment assessment

Be factual, professional, and concise."""

    return call_llm(prompt)


if __name__ == "__main__":
    print("[Task 3] NMF Topics:")
    topics = get_topics_nmf()
    for topic, words in topics.items():
        print(f"  {topic}: {', '.join(words)}")

    print("\n[Dashboard] Topic distribution:")
    dist = get_topic_distribution()
    for theme, stats in dist.items():
        print(f"  {theme}: {stats}")