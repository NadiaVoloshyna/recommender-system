import pandas as pd
from config.paths import FULL_TRAIN_FEATURES, SAMPLED_TRAIN_FEATURES, VAL_FEATURES, TRACK_EMBEDDINGS
from training.train_baseline import train_baseline
from training.train_nn import train_nn
import matplotlib.pyplot as plt


def run_training_pipeline():
    # Load data
    full_train_features = pd.read_parquet(FULL_TRAIN_FEATURES)
    sampled_train_features = pd.read_parquet(SAMPLED_TRAIN_FEATURES)
    val_features = pd.read_parquet(VAL_FEATURES)

    # Train baseline
    baseline_model, baseline_scaler, baseline_metrics, baseline_source = train_baseline(
        full_train_features,
        sampled_train_features,
        val_features
    )
    print(f"\nSelected baseline: {baseline_source}\n")

    # Train NN
    nn_model, nn_metrics = train_nn(sampled_train_features, val_features, baseline_scaler)

    comparison = pd.DataFrame({
        "baseline": [
            baseline_metrics["auc"],
            baseline_metrics["precision@10"],
            baseline_metrics["recall@10"],
            baseline_metrics["ndcg@10"],
            baseline_metrics["precision@20"],
            baseline_metrics["recall@20"],
            baseline_metrics["ndcg@20"],
        ],
        "nn": [
            nn_metrics["auc"],
            nn_metrics["precision@10"],
            nn_metrics["recall@10"],
            nn_metrics["ndcg@10"],
            nn_metrics["precision@20"],
            nn_metrics["recall@20"],
            nn_metrics["ndcg@20"],
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

    comparison.plot(kind="bar", figsize=(12, 6))
    plt.title("Baseline vs Neural Network")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_training_pipeline()

