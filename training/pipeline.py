import pandas as pd
from config.paths import FULL_TRAIN_FEATURES, SAMPLED_TRAIN_FEATURES, VAL_FEATURES, TEST_FEATURES
from training.train_baseline import train_baseline
from training.train_nn import train_nn, train_final_nn, make_tensors, evaluate
from training.train_gbm import train_gbm
from training.utils import plot_model_comparison
from torch.utils.data import TensorDataset, DataLoader


def run_training_pipeline():
    # Load data
    full_train_features = pd.read_parquet(FULL_TRAIN_FEATURES)
    sampled_train_features = pd.read_parquet(SAMPLED_TRAIN_FEATURES)
    val_features = pd.read_parquet(VAL_FEATURES)
    test_features = pd.read_parquet(TEST_FEATURES)

    # ==================== BASELINE ====================
    baseline_model, baseline_scaler, baseline_metrics, baseline_source = train_baseline(
        full_train_features,
        sampled_train_features,
        val_features
    )
    print(f"\nSelected baseline: {baseline_source}\n")

    # ==================== NEURAL NETWORK ====================
    print("====== Training Neural Network model ======\n")
    nn_model, nn_metrics, best_epoch, criterion, device = train_nn(
        sampled_train_features,
        val_features,
        baseline_scaler
    )
    print(f"\nSelected epoch: {best_epoch}")
    print(f"Best NDCG@10: {nn_metrics['ndcg@10']:.4f}\n")

    # ==================== LightGBM ====================
    gbm_model, gbm_metrics = train_gbm(full_train_features, val_features)

    # ==================== COMPARISON ====================
    comparison = {
        "baseline": baseline_metrics,
        "neural network": nn_metrics,
        "lightGBM": gbm_metrics
    }
    comparison_df = pd.DataFrame(comparison)

    print("\n====== Comparison of Logistic Regression Baseline, Neural Network, and LightGBM ======\n")
    print(comparison_df.to_string(float_format=lambda x: f"{x:.4f}"))

    plot_model_comparison(comparison, "Logistic Regression vs Neural Network vs LightGBM")

    # ==================== FINAL MODEL ====================
    print("\n====== Training Final Model ======")
    combined_features = pd.concat([sampled_train_features, val_features], ignore_index=True)

    final_model = train_final_nn(
        train_features=combined_features,
        scaler=baseline_scaler,
        epochs=best_epoch
    )

    test_tensors = make_tensors(test_features, baseline_scaler)
    test_loader = DataLoader(TensorDataset(*test_tensors), batch_size=256, shuffle=False)

    test_loss, test_metrics = evaluate(
        final_model,
        test_loader,
        criterion,
        device,
        test_features
    )

    # ==================== COMPARISON ====================
    comparison = {
        "baseline": baseline_metrics,
        "neural network": nn_metrics,
        "lightGBM": gbm_metrics,
        "final model": test_metrics
    }
    comparison_df = pd.DataFrame(comparison)

    print("\n====== Comparison of Logistic Regression Baseline, Neural Network, LightGBM, and Final Model ======\n")
    print(comparison_df.to_string(float_format=lambda x: f"{x:.4f}"))

    plot_model_comparison(comparison, "Logistic Regression vs Neural Network vs LightGBM vs Final Model")


if __name__ == "__main__":
    run_training_pipeline()

