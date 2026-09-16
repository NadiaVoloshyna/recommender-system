import pandas as pd
from config.paths import FULL_TRAIN_FEATURES, SAMPLED_TRAIN_FEATURES, VAL_FEATURES
from training.train_baseline import train_baseline

# Load data
full_train_features = pd.read_parquet(FULL_TRAIN_FEATURES)
sampled_train_features = pd.read_parquet(SAMPLED_TRAIN_FEATURES)
val_features = pd.read_parquet(VAL_FEATURES)


def run_training_pipeline():
    baseline_model, baseline_scaler, baseline_metrics = train_baseline(
        full_train_features,
        sampled_train_features,
        val_features
    )

    # nn_model, nn_metrics = train_nn(...)

    """
    comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Neural Network"
    ],
    "AUC": [
        baseline_metrics["sampled"]["auc"],
        nn_metrics["auc"]
    ],
    "Precision@10": [
        baseline_metrics["sampled"]["precision@10"],
        nn_metrics["precision@10"]
    ],
    "Recall@10": [
        baseline_metrics["sampled"]["recall@10"],
        nn_metrics["recall@10"]
    ],
    "NDCG@10": [
        baseline_metrics["sampled"]["ndcg@10"],
        nn_metrics["ndcg@10"]
    ],
    "Precision@20": [
        baseline_metrics["sampled"]["precision@20"],
        nn_metrics["precision@20"]
    ],
    "Recall@20": [
        baseline_metrics["sampled"]["recall@20"],
        nn_metrics["recall@20"]
    ],
    "NDCG@20": [
        baseline_metrics["sampled"]["ndcg@20"],
        nn_metrics["ndcg@20"]
    ]
})

print(comparison)
    """


if __name__ == "__main__":
    run_training_pipeline()

