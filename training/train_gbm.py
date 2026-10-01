import pandas as pd
from sklearn.metrics import roc_auc_score
from training.evaluation import evaluate_ranker
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from features.utils import validate_columns


def train_gbm(
        full_train_features: pd.DataFrame,
        val_features: pd.DataFrame
) -> tuple[LGBMClassifier, dict]:
    """
    Trains a LightGBM binary classifier to predict user–track relevance, uses the predicted probabilities
    as recommendation scores, and evaluates both prediction quality (AUC) and recommendation quality
    (Precision/Recall/NDCG at 10 and 20).
    Early stopping is used with the validation set to determine the best number of boosting iterations.
    :param full_train_features: training dataset containing the engineered user-track features and binary
        labels, including the original class distribution (pd.DataFrame)
    :param val_features: validation dataset containing the engineered user-track features. The validation
        labels are used to evaluate both classification and ranking performance (pd.DataFrame)
    :return: tuple[LGBMClassifier, dict]:
        model: the trained LightGBM binary classifier (LGBMClassifier)
        metrics: dictionary containing AUC, Precision@10, Recall@10, NDCG@10, Precision@20, Recall@20,
        and NDCG@20 (dict)
    """
    required_columns = ["user_id", "track_id", "label"]
    validate_columns(full_train_features, required_columns, "full_train_features")
    validate_columns(val_features, required_columns, "val_features")

    if full_train_features["label"].nunique() < 2:
        raise ValueError("Training data must contain both classes.")

    if val_features["label"].nunique() < 2:
        raise ValueError("Validation data must contain both classes.")

    X_train = full_train_features.drop(columns=["user_id", "track_id", "label"])
    y_train = full_train_features["label"]

    X_val = val_features.drop(columns=["user_id", "track_id", "label"])
    y_val = val_features["label"]

    model = LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        importance_type="gain",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        callbacks=[early_stopping(50), log_evaluation(50)]
    )

    importance = (pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False))
    # print(importance.to_string(index=False))

    val_probs = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)

    val_results = val_features[["user_id", "track_id", "label"]].copy()
    val_results["score"] = val_probs
    ranking_results = evaluate_ranker(val_results, ks=[10, 20])

    metrics = {
        "auc": val_auc,
        "precision@10": ranking_results[10]["precision"],
        "recall@10": ranking_results[10]["recall"],
        "ndcg@10": ranking_results[10]["ndcg"],
        "precision@20": ranking_results[20]["precision"],
        "recall@20": ranking_results[20]["recall"],
        "ndcg@20": ranking_results[20]["ndcg"]
    }

    return model, metrics



