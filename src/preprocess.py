# src/preprocess.py
"""
NLP preprocessing pipeline for patient feedback.
Handles: cleaning, tokenization, stopword removal, lemmatization, TF-IDF.
Dataset columns: Feedback, Sentiment, Theme, Satisfaction, Readmission
"""
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
from config import DATA_PATH, TEXT_COL, LABEL_COL, MODEL_PATH

# Download required NLTK data on first run
nltk.download("stopwords",    quiet=True)
nltk.download("wordnet",      quiet=True)
nltk.download("punkt",        quiet=True)
nltk.download("punkt_tabbed", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Clean a single feedback string.
    Steps: lowercase, remove punctuation, remove stopwords, lemmatize.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)


def load_and_preprocess(data_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the Excel dataset and apply text cleaning.
    Returns a DataFrame with a new 'cleaned_text' column.
    """
    df = pd.read_excel(data_path)
    print(f"[Preprocess] Loaded {len(df)} rows | columns: {df.columns.tolist()}")
    df["cleaned_text"] = df[TEXT_COL].apply(clean_text)
    print(f"[Preprocess] Text cleaning complete.")
    return df


def build_features(
    df: pd.DataFrame,
    max_features: int = 5000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Vectorize cleaned text with TF-IDF and split into train/test sets.

    Returns:
        X_train, X_test, y_train, y_test, vectorizer
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["cleaned_text"])
    y = df[LABEL_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[Preprocess] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # Save vectorizer alongside the model
    vectorizer_path = MODEL_PATH.replace(".pkl", "_vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path)
    print(f"[Preprocess] Vectorizer saved to {vectorizer_path}")

    return X_train, X_test, y_train, y_test, vectorizer


if __name__ == "__main__":
    df = load_and_preprocess()
    build_features(df)
    print(df[["Feedback", "cleaned_text", "Sentiment", "Theme"]].head(3))