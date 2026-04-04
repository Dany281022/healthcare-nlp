# src/train.py
"""
Model training with MLflow experiment tracking (local file mode).
Trains a sentiment classifier on patient feedback (Sentiment: 0/1).
"""
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, roc_auc_score
)
from config import (
    MODEL_PATH, MLFLOW_TRACKING_URI, EXPERIMENT_NAME, DATA_PATH
)
from src.preprocess import load_and_preprocess, build_features


def train(model_type: str = "svm") -> dict:
    """
    Train a sentiment classifier and log everything to MLflow.

    Args:
        model_type: 'svm' or 'lr' (logistic regression)
    Returns:
        dict with accuracy, f1, auc metrics
    """
    # ── Setup ──────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Local file mode — no MLflow server needed
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── Data ───────────────────────────────────────────
    df = load_and_preprocess(DATA_PATH)
    X_train, X_test, y_train, y_test, vectorizer = build_features(df)

    # ── Model selection ────────────────────────────────
    if model_type == "svm":
        model = LinearSVC(max_iter=2000, C=1.0)
    else:
        model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")

    # ── Train + MLflow run ─────────────────────────────
    with mlflow.start_run(run_name=f"sentiment_{model_type}"):

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ── Metrics ────────────────────────────────────
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")

        auc = None
        if model_type == "lr":
            proba = model.predict_proba(X_test)[:, 1]
            auc   = roc_auc_score(y_test, proba)

        # ── Log to MLflow ──────────────────────────────
        mlflow.log_param("model_type",   model_type)
        mlflow.log_param("data_rows",    len(df))
        mlflow.log_param("max_features", 5000)
        mlflow.log_metric("accuracy",    acc)
        mlflow.log_metric("f1_score",    f1)
        if auc:
            mlflow.log_metric("auc", auc)

        mlflow.sklearn.log_model(model, "model")

        # ── Console output ─────────────────────────────
        print(f"\n[Train] Model    : {model_type.upper()}")
        print(f"[Train] Accuracy : {acc:.4f}")
        print(f"[Train] F1 Score : {f1:.4f}")
        if auc:
            print(f"[Train] AUC      : {auc:.4f}")
        print("\n[Train] Classification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=["Negative", "Positive"]
        ))

        # ── Save model locally ─────────────────────────
        joblib.dump(model, MODEL_PATH)
        print(f"[Train] Model saved to {MODEL_PATH}")

        metrics = {"accuracy": acc, "f1_score": f1, "auc": auc}

    return metrics


if __name__ == "__main__":
    train(model_type="svm")
    