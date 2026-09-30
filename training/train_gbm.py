import pandas as pd
from sklearn.metrics import roc_auc_score
from training.evaluation import evaluate_ranker
from lightgbm import LGBMClassifier, early_stopping, log_evaluation


def train_gbm(full_train_features: pd.DataFrame, val_features: pd.DataFrame) -> tuple[LGBMClassifier, dict]:
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

    train_probs = model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_probs)
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Best iteration: {model.best_iteration_}")
    importance = (pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False))
    print(importance.to_string(index=False))

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



