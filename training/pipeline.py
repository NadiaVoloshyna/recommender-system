import pandas as pd
from config.paths import FULL_TRAIN_FEATURES, SAMPLED_TRAIN_FEATURES, VAL_FEATURES
from training.train_baseline import train_baseline
from training.train_nn import train_nn

# Load data
full_train_features = pd.read_parquet(FULL_TRAIN_FEATURES)
sampled_train_features = pd.read_parquet(SAMPLED_TRAIN_FEATURES)
val_features = pd.read_parquet(VAL_FEATURES)


def run_training_pipeline():
    baseline_model, baseline_scaler, baseline_metrics, baseline_source = train_baseline(
        full_train_features,
        sampled_train_features,
        val_features
    )

    print(f"\nSelected baseline: {baseline_source}")

    nn_model, nn_metrics = train_nn(sampled_train_features, val_features, baseline_scaler)


if __name__ == "__main__":
    run_training_pipeline()

