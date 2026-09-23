import pandas as pd
from pprint import pprint
from training.preprocessing import fit_transform_features, transform_features
from training.evaluation import evaluate_ranker
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from training.utils import plot_baseline_comparison


def train_baseline(
        full_train_features: pd.DataFrame,
        sampled_train_features: pd.DataFrame,
        val_features: pd.DataFrame
) -> tuple[LogisticRegression, StandardScaler, dict, str]:
    """
    Train and compare two logistic-regression baseline models.
    The first model is trained on the full, highly imbalanced training dataset using class weighting.
    The second model is trained on a negatively sampled training dataset. Both models are evaluated on
    the same validation dataset using AUC, Precision@K, Recall@K, and NDCG@K.
    The model with the higher NDCG@10 is selected and returned, together with its feature scaler and
    the evaluation metrics for both baseline models.
    :param full_train_features: training dataset containing the engineered ranking features and binary labels,
    including the original class distribution (pd.DataFrame)
    :param sampled_train_features: training dataset containing the engineered ranking features and binary
    labels after negative sampling (pd.DataFrame)
    :param val_features: validation dataset containing the same engineered ranking features
    and binary labels (pd.DataFrame)
    :return:
        tuple[LogisticRegression, StandardScaler, dict, str]:
            The selected logistic-regression model, the feature scaler fitted
            on the corresponding training dataset, a dictionary containing the
            evaluation metrics for both baseline models, and a string identifying
            the selected model ("full" or "sampled").
    """
    if not isinstance(full_train_features, pd.DataFrame):
        raise TypeError("full_train_features must be a pandas DataFrame")

    if not isinstance(sampled_train_features, pd.DataFrame):
        raise TypeError("sampled_train_features must be a pandas DataFrame")

    if not isinstance(val_features, pd.DataFrame):
        raise TypeError("val_features must be a pandas DataFrame")

    # Model 1: Full, highly imbalanced dataset
    model_full = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    X_train_full, scaler_full = fit_transform_features(full_train_features)
    y_train_full = full_train_features["label"]

    model_full.fit(X_train_full, y_train_full)

    X_val_full = transform_features(val_features, scaler_full)
    y_val = val_features["label"]

    val_pred_full = model_full.predict_proba(X_val_full)[:, 1]

    val_results_full = val_features[["user_id", "track_id", "label"]].copy()
    val_results_full["score"] = val_pred_full
    results_full = evaluate_ranker(val_results_full, ks=[10, 20])

    auc_full = roc_auc_score(y_val, val_pred_full)

    # Model 2: Negatively sampled dataset
    model_sampled = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    X_train_sampled, scaler_sampled = fit_transform_features(sampled_train_features)
    y_train_sampled = sampled_train_features["label"]

    model_sampled.fit(X_train_sampled, y_train_sampled)

    X_val_sampled = transform_features(val_features, scaler_sampled)
    y_val = val_features["label"]

    val_pred_sampled = model_sampled.predict_proba(X_val_sampled)[:, 1]

    val_results_sampled = val_features[["user_id", "track_id", "label"]].copy()
    val_results_sampled["score"] = val_pred_sampled
    results_sampled = evaluate_ranker(val_results_sampled, ks=[10, 20])

    auc_sampled = roc_auc_score(y_val, val_pred_sampled)

    # Store metrics for both models
    metrics = {
        "full": {
            "auc": auc_full,
            "precision@10": results_full[10]["precision"],
            "recall@10": results_full[10]["recall"],
            "ndcg@10": results_full[10]["ndcg"],
            "precision@20": results_full[20]["precision"],
            "recall@20": results_full[20]["recall"],
            "ndcg@20": results_full[20]["ndcg"]
        },
        "sampled": {
            "auc": auc_sampled,
            "precision@10": results_sampled[10]["precision"],
            "recall@10": results_sampled[10]["recall"],
            "ndcg@10": results_sampled[10]["ndcg"],
            "precision@20": results_sampled[20]["precision"],
            "recall@20": results_sampled[20]["recall"],
            "ndcg@20": results_sampled[20]["ndcg"]
        }
    }

    # plot_baseline_comparison(metrics)

    # Select model based on NDCG@10
    if metrics["full"]["ndcg@10"] >= metrics["sampled"]["ndcg@10"]:
        return model_full, scaler_full, metrics["full"], "full"
    else:
        return model_sampled, scaler_sampled, metrics["sampled"], "sampled"


"""
comparison = pd.DataFrame({
        "full": [
            auc_full,
            results_full[10]["precision"],
            results_full[10]["recall"],
            results_full[10]["ndcg"],
            results_full[20]["precision"],
            results_full[20]["recall"],
            results_full[20]["ndcg"]
        ],
        "sampled": [
            auc_sampled,
            results_sampled[10]["precision"],
            results_sampled[10]["recall"],
            results_sampled[10]["ndcg"],
            results_sampled[20]["precision"],
            results_sampled[20]["recall"],
            results_sampled[20]["ndcg"]
        ],
    }, index=[
        "AUC",
        "Precision@10",
        "Recall@10",
        "NDCG@10",
        "Precision@20",
        "Recall@20",
        "NDCG@20",
    ])

    print(comparison)
"""

