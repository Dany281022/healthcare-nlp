# src/train.py
"""
Model training — 3 tâches du projet Cambrian Healthcare NLP.

Task 1 : Sentiment classification  (supervisé  — LogisticRegression / LinearSVC)
         Target : Sentiment (0 = Negative, 1 = Positive)
         Métriques : accuracy, f1_macro, f1_weighted, auc, recall_neg, precision_neg

Task 2 : Theme classification      (supervisé  — LogisticRegression multiclasse)
         Target : Theme (communication, wait_time, medication, discharge)
         Métriques : accuracy, f1_macro

Task 3 : NMF Topic Modeling        (non supervisé — NMF sur matrice TF-IDF)
         Pas de labels. Extrait 4 topics latents + top mots par topic.
         Métriques : reconstruction_error (loggué dans MLflow)

Pipeline mathématique : TF-IDF → modèles linéaires (LR, SVC, NMF)
Tout est tracké dans MLflow sous le même experiment.
"""
import os
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import NMF
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    classification_report,
)
from config import (
    MODEL_PATH,
    SENTIMENT_MODEL_PATH,
    THEME_MODEL_PATH,
    NMF_MODEL_PATH,
    TFIDF_PATH,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    DATA_PATH,
    NMF_N_COMPONENTS,
    NMF_MAX_ITER,
    NMF_TOP_WORDS,
    THEMES,
)
from src.preprocess import load_and_preprocess, build_features


# ── Shared setup ───────────────────────────────────────
def _setup():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


# ══════════════════════════════════════════════════════
# TASK 1 — Sentiment Classification
# ══════════════════════════════════════════════════════
def train_sentiment(
    X_train, X_test,
    y_train, y_test,
    model_type: str = "svm",
) -> dict:
    """
    Task 1 : binary sentiment classifier (Positive / Negative).

    Args:
        model_type: 'svm' (LinearSVC) or 'lr' (LogisticRegression)
    Returns:
        dict of all tracked metrics
    """
    if model_type == "svm":
        model = LinearSVC(max_iter=2000, C=1.0)
    else:
        model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")

    with mlflow.start_run(
        run_name=f"task1_sentiment_{model_type}",
        tags={"task": "1", "type": "sentiment_classification", "model": model_type},
    ):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc           = accuracy_score(y_test, y_pred)
        f1_weighted   = f1_score(y_test, y_pred, average="weighted")
        f1_macro      = f1_score(y_test, y_pred, average="macro")
        recall_neg    = recall_score(y_test, y_pred, pos_label=0)
        precision_neg = precision_score(y_test, y_pred, pos_label=0)
        recall_pos    = recall_score(y_test, y_pred, pos_label=1)
        precision_pos = precision_score(y_test, y_pred, pos_label=1)

        if model_type == "svm":
            scores = model.decision_function(X_test)
            auc = roc_auc_score(y_test, scores)
        else:
            proba = model.predict_proba(X_test)[:, 1]
            auc   = roc_auc_score(y_test, proba)

        mlflow.log_param("model_type",    model_type)
        mlflow.log_param("task",          "sentiment_classification")
        mlflow.log_metric("accuracy",     acc)
        mlflow.log_metric("f1_weighted",  f1_weighted)
        mlflow.log_metric("f1_macro",     f1_macro)
        mlflow.log_metric("auc",          auc)
        mlflow.log_metric("recall_neg",   recall_neg)
        mlflow.log_metric("precision_neg",precision_neg)
        mlflow.log_metric("recall_pos",   recall_pos)
        mlflow.log_metric("precision_pos",precision_pos)
        mlflow.set_tag("primary_metric",  "recall_neg")
        mlflow.sklearn.log_model(model, "sentiment_model")

        print(f"\n{'='*50}")
        print(f"TASK 1 — Sentiment Classification ({model_type.upper()})")
        print(f"{'='*50}")
        print(f"  Accuracy      : {acc:.4f}")
        print(f"  F1 Weighted   : {f1_weighted:.4f}")
        print(f"  F1 Macro      : {f1_macro:.4f}")
        print(f"  AUC           : {auc:.4f}")
        print(f"  Recall  (neg) : {recall_neg:.4f}  <- primary metric")
        print(f"  Precision(neg): {precision_neg:.4f}")
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Negative","Positive"]))

        joblib.dump(model, SENTIMENT_MODEL_PATH)
        print(f"  Model saved to {SENTIMENT_MODEL_PATH}")

        return {
            "accuracy": acc, "f1_weighted": f1_weighted, "f1_macro": f1_macro,
            "auc": auc, "recall_neg": recall_neg, "precision_neg": precision_neg,
        }


# ══════════════════════════════════════════════════════
# TASK 2 — Theme Classification
# ══════════════════════════════════════════════════════
def train_theme(
    X_train, X_test,
    y_train, y_test,
) -> dict:
    """
    Task 2 : multiclass theme classifier.
    Predicts one of: communication, wait_time, medication, discharge.

    Uses LogisticRegression with multinomial strategy.
    Same TF-IDF matrix as Task 1 — no refit needed.
    """
    if y_train is None or y_test is None:
        print("\n[Task 2] Skipped — Theme column not found in dataset.")
        return {}

    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        multi_class="multinomial",
    )

    with mlflow.start_run(
        run_name="task2_theme_classification",
        tags={"task": "2", "type": "theme_classification", "model": "logreg"},
    ):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc      = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_w     = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_param("model_type",   "logistic_regression_multinomial")
        mlflow.log_param("task",         "theme_classification")
        mlflow.log_param("num_classes",  len(set(y_train)))
        mlflow.log_metric("accuracy",    acc)
        mlflow.log_metric("f1_macro",    f1_macro)
        mlflow.log_metric("f1_weighted", f1_w)
        mlflow.sklearn.log_model(model, "theme_model")

        print(f"\n{'='*50}")
        print("TASK 2 — Theme Classification (LogReg Multinomial)")
        print(f"{'='*50}")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  F1 Macro  : {f1_macro:.4f}")
        print(f"  F1 Weighted: {f1_w:.4f}")
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=sorted(set(y_test))))

        joblib.dump(model, THEME_MODEL_PATH)
        print(f"  Model saved to {THEME_MODEL_PATH}")

        return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_w}


# ══════════════════════════════════════════════════════
# TASK 3 — NMF Topic Modeling
# ══════════════════════════════════════════════════════
def train_nmf(X_train, vectorizer) -> dict:
    """
    Task 3 : NMF topic modeling (unsupervised).

    Applies NMF on the full TF-IDF train matrix.
    Extracts NMF_N_COMPONENTS latent topics.
    Logs top words per topic to MLflow as artifact.

    No labels required — purely unsupervised.
    """
    model = NMF(
        n_components=NMF_N_COMPONENTS,
        max_iter=NMF_MAX_ITER,
        random_state=42,
    )

    with mlflow.start_run(
        run_name="task3_nmf_topic_modeling",
        tags={"task": "3", "type": "topic_modeling", "model": "nmf"},
    ):
        W = model.fit_transform(X_train)  # document-topic matrix
        H = model.components_             # topic-term matrix

        recon_error = model.reconstruction_err_

        # Extract top words per topic
        feature_names = vectorizer.get_feature_names_out()
        topics = {}
        print(f"\n{'='*50}")
        print(f"TASK 3 — NMF Topic Modeling ({NMF_N_COMPONENTS} topics)")
        print(f"{'='*50}")
        print(f"  Reconstruction error : {recon_error:.4f}")
        print(f"  Document-topic matrix : {W.shape}")

        for topic_idx, topic in enumerate(H):
            top_indices = topic.argsort()[-NMF_TOP_WORDS:][::-1]
            top_words   = [feature_names[i] for i in top_indices]
            topic_key   = f"topic_{topic_idx + 1}"
            topics[topic_key] = top_words
            print(f"\n  Topic {topic_idx + 1}: {', '.join(top_words)}")

        mlflow.log_param("n_components",       NMF_N_COMPONENTS)
        mlflow.log_param("max_iter",           NMF_MAX_ITER)
        mlflow.log_param("top_words_per_topic",NMF_TOP_WORDS)
        mlflow.log_metric("reconstruction_error", recon_error)

        # Log topics as JSON artifact
        topics_path = MODEL_PATH.replace("sentiment_model.pkl", "nmf_topics.json")
        with open(topics_path, "w") as f:
            json.dump(topics, f, indent=2)
        mlflow.log_artifact(topics_path, "topics")
        print(f"\n  Topics saved to {topics_path}")

        joblib.dump(model, NMF_MODEL_PATH)
        print(f"  NMF model saved to {NMF_MODEL_PATH}")

        return {"reconstruction_error": recon_error, "topics": topics}


# ══════════════════════════════════════════════════════
# MAIN — Run all 3 tasks
# ══════════════════════════════════════════════════════
def train(model_type: str = "svm") -> dict:
    """
    Run the full training pipeline : Task 1 + Task 2 + Task 3.

    Single data load, single vectorizer fit.
    All 3 tasks share the same TF-IDF representation.
    """
    _setup()

    # ── Data ───────────────────────────────────────────
    df = load_and_preprocess(DATA_PATH)
    X_train, X_test, y_s_train, y_s_test, y_t_train, y_t_test, vectorizer = build_features(df)

    print(f"\n[Train] Dataset     : {DATA_PATH}")
    print(f"[Train] Total rows  : {len(df)}")
    print(f"[Train] Train/Test  : {X_train.shape[0]} / {X_test.shape[0]}")
    print(f"[Train] Vocab size  : {X_train.shape[1]}")

    results = {}

    # ── Task 1 ─────────────────────────────────────────
    results["task1"] = train_sentiment(
        X_train, X_test, y_s_train, y_s_test, model_type=model_type
    )

    # ── Task 2 ─────────────────────────────────────────
    results["task2"] = train_theme(
        X_train, X_test, y_t_train, y_t_test
    )

    # ── Task 3 ─────────────────────────────────────────
    results["task3"] = train_nmf(X_train, vectorizer)

    print(f"\n{'='*50}")
    print("TRAINING COMPLETE — 3 tasks logged to MLflow")
    print(f"{'='*50}")
    return results


if __name__ == "__main__":
    train(model_type="svm")
