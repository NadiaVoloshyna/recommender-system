import pandas as pd
from config.paths import FULL_TRAIN_FEATURES, SAMPLED_TRAIN_FEATURES, VAL_FEATURES
from training.train_baseline import train_baseline
from training.train_nn import train_nn
from training.utils import plot_model_comparison


def run_training_pipeline():
    # Load data
    full_train_features = pd.read_parquet(FULL_TRAIN_FEATURES)
    sampled_train_features = pd.read_parquet(SAMPLED_TRAIN_FEATURES)
    val_features = pd.read_parquet(VAL_FEATURES)

    # ==================== BASELINE ====================
    baseline_model, baseline_scaler, baseline_metrics, baseline_source = train_baseline(
        full_train_features,
        sampled_train_features,
        val_features
    )
    print(f"\nSelected baseline: {baseline_source}\n")

    # ==================== NEURAL NETWORK ====================
    print("====== Training Neural Network model ======\n")
    nn_model, nn_metrics = train_nn(
        sampled_train_features,
        val_features,
        baseline_scaler
    )

    # ==================== COMPARISON ====================
    comparison = {
        "baseline": baseline_metrics,
        "neural network": nn_metrics,
    }
    comparison_df = pd.DataFrame(comparison)

    print("\n====== Comparison of Baseline and Neural Network ======\n")
    print(comparison_df.to_string(float_format=lambda x: f"{x:.4f}"))

    plot_model_comparison(comparison, "Baseline and Neural Network Comparison")


if __name__ == "__main__":
    run_training_pipeline()

