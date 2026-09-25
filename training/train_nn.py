import pandas as pd
import torch
import torch.nn as nn
import copy
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from training.preprocessing import transform_features
from training.evaluation import evaluate_ranker
from training.utils import plot_training_history

BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 10


class NN(nn.Module):
    def __init__(self, n_numeric: int):
        super().__init__()

        self.ranking_mlp = nn.Sequential(
            nn.Linear(n_numeric, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, numeric_features):
        return self.ranking_mlp(numeric_features).squeeze(1)


def make_tensors(df: pd.DataFrame, scaler: StandardScaler):
    numeric_features = transform_features(df, scaler)
    numeric_features = torch.tensor(numeric_features.to_numpy(), dtype=torch.float32)

    labels = torch.tensor(df["label"].to_numpy(), dtype=torch.float32)

    return numeric_features, labels


def evaluate(model, loader, criterion, device, val_features):
    model.eval()

    total_val_loss = 0
    logits_list = []
    targets_list = []

    with torch.no_grad():
        for numeric, targets in loader:
            numeric = numeric.to(device)
            targets = targets.to(device)

            logits = model(numeric)
            loss = criterion(logits, targets)
            total_val_loss += (loss.item() * len(targets))

            logits_list.append(logits.cpu())
            targets_list.append(targets.cpu())

    logits = torch.cat(logits_list)
    probs = torch.sigmoid(logits)
    targets = torch.cat(targets_list)

    avg_val_loss = total_val_loss / len(targets)
    auc = roc_auc_score(targets.numpy(), probs.numpy())

    val_results = val_features[["user_id", "track_id", "label"]].copy()
    val_results["score"] = probs.numpy()

    ranking_results = evaluate_ranker(val_results, ks=[10, 20])

    metrics = {
        "auc": auc,
        "precision@10": ranking_results[10]["precision"],
        "recall@10": ranking_results[10]["recall"],
        "ndcg@10": ranking_results[10]["ndcg"],
        "precision@20": ranking_results[20]["precision"],
        "recall@20": ranking_results[20]["recall"],
        "ndcg@20": ranking_results[20]["ndcg"]
    }

    return avg_val_loss, metrics


def train_nn(
        sampled_train_features: pd.DataFrame,
        val_features: pd.DataFrame,
        scaler: StandardScaler
) -> tuple[NN, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tensors = make_tensors(sampled_train_features, scaler)
    val_tensors = make_tensors(val_features, scaler)
    n_numeric = train_tensors[0].shape[1]

    train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_tensors), batch_size=BATCH_SIZE, shuffle=False)

    model = NN(n_numeric=n_numeric).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_ndcg = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = None
    history = {
        "train_loss": [],
        "val_loss": [],
        "auc": [],
        "ndcg@10": [],
        "ndcg@20": [],
    }

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        for numeric, targets in train_loader:
            numeric = numeric.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            logits = model(numeric)
            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            total_train_loss += (loss.item() * len(targets))

        avg_train_loss = total_train_loss / len(train_tensors[0])

        val_loss, metrics = evaluate(model, val_loader, criterion, device, val_features)
        if metrics["ndcg@10"] > best_ndcg:
            best_ndcg = metrics["ndcg@10"]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics.copy()

        print(
            f"Epoch {epoch + 1}/{EPOCHS}   "
            f"train_loss={avg_train_loss:.4f}   "
            f"val_loss={val_loss:.4f}   "
            f"auc={metrics['auc']:.4f}   "
            f"precision@10={metrics['precision@10']:.4f}   "
            f"recall@10={metrics['recall@10']:.4f}   "
            f"ndcg@10={metrics['ndcg@10']:.4f}   "
            f"precision@20={metrics['precision@20']:.4f}   "
            f"recall@20={metrics['recall@20']:.4f}   "
            f"ndcg@20={metrics['ndcg@20']:.4f}"
        )

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["auc"].append(metrics["auc"])
        history["ndcg@10"].append(metrics["ndcg@10"])
        history["ndcg@20"].append(metrics["ndcg@20"])

    # Restore best model
    model.load_state_dict(best_state)
    plot_training_history(history)

    return model, best_metrics






