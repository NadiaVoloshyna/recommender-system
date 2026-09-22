import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from training.preprocessing import transform_features
from training.evaluation import evaluate_ranker
from pprint import pprint

EMBEDDING_DIM = 32
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 10
NUMERIC_COLUMNS = 17


class NN(nn.Module):
    def __init__(self, n_users, n_tracks, n_numeric):
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, EMBEDDING_DIM)
        self.track_embedding = nn.Embedding(n_tracks, EMBEDDING_DIM)

        self.numeric_mlp = nn.Sequential(
            nn.Linear(n_numeric, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.ranking_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_id, track_id, numeric_features):
        user = self.user_embedding(user_id)
        track = self.track_embedding(track_id)
        interaction = user * track
        numeric = self.numeric_mlp(numeric_features)

        x = torch.cat([user, track, interaction, numeric], dim=1)

        return self.ranking_mlp(x).squeeze(1)


def make_tensors(df, scaler, user_to_idx, track_to_idx):
    # Convert string IDs into integer embedding indices
    user_ids = (df["user_id"].map(user_to_idx).fillna(0).astype(np.int64).to_numpy())
    user_ids = torch.tensor(user_ids, dtype=torch.long)
    track_ids = (df["track_id"].map(track_to_idx).fillna(0).astype(np.int64).to_numpy())
    track_ids = torch.tensor(track_ids, dtype=torch.long)

    # Transform numerical features
    numeric_features = transform_features(df, scaler)
    numeric_features = torch.tensor(numeric_features.to_numpy(), dtype=torch.float32)

    labels = torch.tensor(df["label"].to_numpy(), dtype=torch.float32)

    return user_ids, track_ids, numeric_features, labels


def evaluate(model, loader, criterion, device, val_features):
    model.eval()
    total_loss = 0
    logits_list = []
    targets_list = []

    with torch.no_grad():
        for user_ids, track_ids, numeric, targets in loader:
            user_ids = user_ids.to(device)
            track_ids = track_ids.to(device)
            numeric = numeric.to(device)
            targets = targets.to(device)

            logits = model(user_ids, track_ids, numeric)
            loss = criterion(logits, targets)
            total_loss += (loss.item() * len(targets))

            logits_list.append(logits.cpu())
            targets_list.append(targets.cpu())

    logits = torch.cat(logits_list)
    targets = torch.cat(targets_list)

    probs = torch.sigmoid(logits)

    avg_loss = total_loss / len(targets)
    auc = roc_auc_score(targets.numpy(), probs.numpy())

    val_results = val_features[["user_id", "track_id", "label"]].copy()
    val_results["score"] = probs.numpy()

    ranking_results = evaluate_ranker(val_results, ks=[10, 20])

    metrics = {
        "loss": avg_loss,
        "auc": auc,
        "precision@10": ranking_results[10]["precision"],
        "recall@10": ranking_results[10]["recall"],
        "ndcg@10": ranking_results[10]["ndcg"],
        "precision@20": ranking_results[20]["precision"],
        "recall@20": ranking_results[20]["recall"],
        "ndcg@20": ranking_results[20]["ndcg"],
        "logits": logits,
        "probs": probs,
        "targets": targets
    }

    return metrics


def train_nn(
        sampled_train_features: pd.DataFrame,
        val_features: pd.DataFrame,
        scaler: StandardScaler
) -> tuple[NN, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create id mappings
    user_to_idx = {x: i + 1 for i, x in enumerate(sampled_train_features["user_id"].unique())}
    track_to_idx = {x: i + 1 for i, x in enumerate(sampled_train_features["track_id"].unique())}

    train_tensors = make_tensors(sampled_train_features, scaler, user_to_idx, track_to_idx)
    val_tensors = make_tensors(val_features, scaler, user_to_idx, track_to_idx)

    train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_tensors), batch_size=BATCH_SIZE, shuffle=False)

    model = NN(
        n_users=len(user_to_idx) + 1,
        n_tracks=len(track_to_idx) + 1,
        n_numeric=NUMERIC_COLUMNS,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for user_ids, track_ids, numeric, targets in train_loader:
            user_ids = user_ids.to(device)
            track_ids = track_ids.to(device)
            numeric = numeric.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            logits = model(user_ids, track_ids, numeric)
            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            train_loss += (loss.item() * len(targets))

        train_loss /= len(train_loader.dataset)

        metrics = evaluate(model, val_loader, criterion, device, val_features)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} "
            f"auc={metrics['auc']:.4f} "
            f"precision@10={metrics['precision@10']:.4f} "
            f"recall@10={metrics['recall@10']:.4f} "
            f"ndcg@10={metrics['ndcg@10']:.4f} "
            f"precision@20={metrics['precision@20']:.4f} "
            f"recall@20={metrics['recall@20']:.4f} "
            f"ndcg@20={metrics['ndcg@20']:.4f}"
        )

    print("\n====== NN Final Metrics ======")
    final_metrics = {
        "auc": metrics['auc'],
        "precision@10": metrics['precision@10'],
        "recall@10": metrics['recall@10'],
        "ndcg@10": metrics['ndcg@10'],
        "precision@20": metrics['precision@20'],
        "recall@20": metrics['recall@20'],
        "ndcg@20": metrics['ndcg@20']
    }
    pprint(final_metrics)

    return model, metrics








