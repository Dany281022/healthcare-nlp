# src/preprocess.py
"""
NLP preprocessing pipeline for patient feedback.

Supports both datasets:
  - data/patient_feedback.csv          (UCI Drug Reviews — texte réel)
  - data/patient_feedback_dataset.xlsx (dataset synthétique Cambrian)
Auto-detection basée sur l'extension du fichier.

Pipeline commun aux 3 tâches :
  Task 1 — Sentiment classification  : TF-IDF + target = Sentiment (0/1)
  Task 2 — Theme classification      : TF-IDF + target = Theme (4 classes)
  Task 3 — NMF Topic modeling        : TF-IDF matrix (non supervisé)

FIX data leakage : split BEFORE vectorizer.fit().
"""
import re
import os
import html
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
from config import DATA_PATH, TEXT_COL, LABEL_COL, THEME_COL, MODEL_PATH, TFIDF_PATH

# Download required NLTK data on first run
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Clean a single feedback string.

    Steps:
    1. Decode HTML entities (&#039; → ') — requis pour drug reviews dataset
    2. Lowercase
    3. Remove digits and punctuation
    4. Remove stopwords
    5. Lemmatize
    6. Drop tokens shorter than 3 chars
    """
    text = html.unescape(str(text))
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 2
    ]
    return " ".join(tokens)


def load_and_preprocess(data_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the dataset (CSV or XLSX) and apply text cleaning.
    Returns a DataFrame with a new 'cleaned_text' column.
    """
    # Auto-detect format
    if data_path.endswith(".xlsx"):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    print(f"[Preprocess] Loaded {len(df)} rows | columns: {df.columns.tolist()}")

    # Validate required columns
    required = [TEXT_COL, LABEL_COL]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[Preprocess] Missing columns: {missing}. "
            f"Found: {df.columns.tolist()}"
        )

    # Drop nulls on required columns
    before = len(df)
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[Preprocess] Dropped {dropped} null rows.")

    # Clean text
    df["cleaned_text"] = df[TEXT_COL].apply(clean_text)

    # Drop rows where cleaning produced empty string
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)

    unique_tokens = len(set(" ".join(df["cleaned_text"]).split()))
    print(f"[Preprocess] Cleaning done. {len(df)} rows | {unique_tokens} unique tokens")

    return df


def build_features(
    df: pd.DataFrame,
    max_features: int = 10000,
    test_size: float  = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Vectorize cleaned text with TF-IDF and split into train/test sets.

    The SAME vectorizer and train/test split are shared across all 3 tasks:
      - Task 1 : y_sentiment (0/1)
      - Task 2 : y_theme     (4 classes)
      - Task 3 : X_train_full (full matrix for NMF, no labels needed)

    Returns:
        X_train, X_test,
        y_s_train, y_s_test,   (Task 1 — sentiment)
        y_t_train, y_t_test,   (Task 2 — theme, None if column missing)
        vectorizer
    """
    y_sentiment = df[LABEL_COL].values
    y_theme     = df[THEME_COL].values if THEME_COL in df.columns else None
    X_text      = df["cleaned_text"].values

    # 1. Split BEFORE vectorisation — prevents data leakage
    indices           = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y_sentiment,
    )

    X_train_text = X_text[train_idx]
    X_test_text  = X_text[test_idx]

    y_s_train = y_sentiment[train_idx]
    y_s_test  = y_sentiment[test_idx]

    y_t_train = y_theme[train_idx] if y_theme is not None else None
    y_t_test  = y_theme[test_idx]  if y_theme is not None else None

    # 2. Fit vectorizer on TRAIN only
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,           # ignore très rares tokens
        sublinear_tf=True,  # log normalization — meilleur pour longs textes
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test  = vectorizer.transform(X_test_text)

    print(f"[Preprocess] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"[Preprocess] TF-IDF vocab size: {X_train.shape[1]}")

    # 3. Save shared vectorizer
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(vectorizer, TFIDF_PATH)
    print(f"[Preprocess] Vectorizer saved to {TFIDF_PATH}")

    # Legacy path — garde api/main.py compatible sans modification
    legacy_path = MODEL_PATH.replace(".pkl", "_vectorizer.pkl")
    joblib.dump(vectorizer, legacy_path)

    return X_train, X_test, y_s_train, y_s_test, y_t_train, y_t_test, vectorizer


if __name__ == "__main__":
    df = load_and_preprocess()
    result = build_features(df)
    X_train = result[0]
    print(f"\n[Preprocess] TF-IDF matrix shape (train): {X_train.shape}")
    print(df[[TEXT_COL, "cleaned_text", LABEL_COL]].head(3))
