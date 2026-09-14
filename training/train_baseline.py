import pandas as pd
from config.paths import FULL_TRAIN_FEATURES, SAMPLED_TRAIN_FEATURES, VAL_FEATURES
from preprocessing import fit_transform_features, transform_features
from evaluation import evaluate_ranker
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Load data
full_train_features = pd.read_parquet(FULL_TRAIN_FEATURES)
sampled_train_features = pd.read_parquet(SAMPLED_TRAIN_FEATURES)
val_features = pd.read_parquet(VAL_FEATURES)


def train_baseline(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


# Train baseline model on the original, highly imbalanced dataset
X_train_full, scaler_full = fit_transform_features(full_train_features)
y_train_full = full_train_features["label"]

X_val = transform_features(val_features, scaler_full)
y_val = val_features["label"]

model = train_baseline(X_train_full, y_train_full)

# Evaluation
val_pred = model.predict_proba(X_val)[:, 1]

auc = roc_auc_score(y_val, val_pred)

val_results = val_features[["user_id", "track_id", "label"]].copy()
val_results["score"] = val_pred

results = evaluate_ranker(val_results, ks=[10, 20])

print("\n=== Original, highly imbalanced dataset ===")
print(f"Validation AUC: {auc:.4f}")

for k, metrics in results.items():
    print(f"\nK = {k}")
    print(f"Precision@{k}: {metrics['precision']:.4f}")
    print(f"Recall@{k}:    {metrics['recall']:.4f}")
    print(f"NDCG@{k}:      {metrics['ndcg']:.4f}")

# Train baseline model on the negatively sampled dataset
X_train_sampled, scaler_sampled = fit_transform_features(sampled_train_features)
y_train_sampled = sampled_train_features["label"]

X_val = transform_features(val_features, scaler_sampled)
y_val = val_features["label"]

model = train_baseline(X_train_sampled, y_train_sampled)

# Evaluation
val_pred = model.predict_proba(X_val)[:, 1]

auc = roc_auc_score(y_val, val_pred)

val_results = val_features[["user_id", "track_id", "label"]].copy()
val_results["score"] = val_pred

results = evaluate_ranker(val_results, ks=[10, 20])

print("\n=== Negatively sampled dataset ===")
print(f"Validation AUC: {auc:.4f}")

for k, metrics in results.items():
    print(f"\nK = {k}")
    print(f"Precision@{k}: {metrics['precision']:.4f}")
    print(f"Recall@{k}:    {metrics['recall']:.4f}")
    print(f"NDCG@{k}:      {metrics['ndcg']:.4f}")

