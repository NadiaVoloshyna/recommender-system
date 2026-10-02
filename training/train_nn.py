import pandas as pd
import torch
import torch.nn as nn
import copy
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from training.preprocessing import fit_transform_features, transform_features
from training.evaluation import evaluate_ranker
from training.utils import plot_training_history
from features.utils import validate_columns
from training.negative_sampling import add_candidate_hardness_scores, sample_negatives_by_hardness
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def make_tensors(df: pd.DataFrame, scaler=None):
    if scaler is None:
        numeric_features, scaler = fit_transform_features(df)
    else:
        numeric_features = transform_features(df, scaler)

    numeric_features = torch.tensor(
        numeric_features.to_numpy(),
        dtype=torch.float32
    )

    labels = torch.tensor(
        df["label"].to_numpy(),
        dtype=torch.float32
    )

    return (numeric_features, labels), scaler


def evaluate(
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        val_features: pd.DataFrame
) -> tuple[float, dict[str, float]]:
    """
    Evaluates a trained neural network on validation data.
    The model is evaluated without updating its parameters. Validation loss and classification performance (AUC)
    are calculated from the model's predictions. The predictions are also combined with user, track, and label
    information to evaluate ranking performance at K=10 and K=20.
    :param model: trained PyTorch neural network to evaluate
    :param val_loader: DataLoader containing validation features and binary target labels (DataLoader)
    :param device: device on which the model and validation tensors are evaluated (torch.device)
    :param val_features: validation dataframe containing `user_id`, `track_id`, and `label` columns (pd.DataFrame)
    :return: tuple[float, dict]:
        avg_val_loss: average validation loss across all validation samples (float)
        metrics: dictionary containing AUC, Precision@10, Recall@10, NDCG@10, Precision@20, Recall@20,
        and NDCG@20 (dict)
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")

    if not isinstance(val_loader, torch.utils.data.DataLoader):
        raise TypeError("loader must be a torch.utils.data.DataLoader")

    if not isinstance(val_features, pd.DataFrame):
        raise TypeError("val_features must be a pandas DataFrame")

    required_columns = ["user_id", "track_id", "label"]
    validate_columns(val_features, required_columns, "val_features")

    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_val_loss = 0
    logits_list = []
    targets_list = []

    with torch.no_grad():
        for numeric, targets in val_loader:
            numeric = numeric.to(DEVICE)
            targets = targets.to(DEVICE)

            logits = model(numeric)
            loss = criterion(logits, targets)
            total_val_loss += (loss.item() * len(targets))

            logits_list.append(logits.cpu())
            targets_list.append(targets.cpu())

    logits = torch.cat(logits_list)
    probs = torch.sigmoid(logits)
    targets = torch.cat(targets_list)
    if len(torch.unique(targets)) < 2:
        raise ValueError(
            "Validation targets must contain both classes to calculate ROC AUC"
        )

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


def train_one_epoch(
        model: NN,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
) -> float:
    model.train()

    total_train_loss = 0.0
    total_samples = 0

    for numeric, targets in train_loader:
        numeric = numeric.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(numeric)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * BATCH_SIZE
        total_samples += BATCH_SIZE

    return total_train_loss / total_samples


def train_nn(
        sampled_train_features: pd.DataFrame,
        val_features: pd.DataFrame
) -> tuple[NN, dict, int]:
    """
    Trains a binary-classification neural network, evaluates its ranking performance on validation data after
    every epoch, saves the model state from the epoch with the highest validation NDCG@10, restores that version
    at the end, and returns the best model and its metrics.
    :param sampled_train_features:training dataset containing the engineered ranking features and binary labels
    after negative sampling (pd.DataFrame)
    :param val_features: validation dataset containing the same engineered ranking features and
    binary labels (pd.DataFrame)
    :return: tuple[NN, dict, int, torch.nn.Module, torch.device]:
        the neural network restored to the epoch with the highest NDCG@10,
        a dictionary containing the evaluation metrics for that model,
        the best epoch.
    """
    if not isinstance(sampled_train_features, pd.DataFrame):
        raise TypeError("sampled_train_features must be a pandas DataFrame")

    if not isinstance(val_features, pd.DataFrame):
        raise TypeError("val_features must be a pandas DataFrame")

    if sampled_train_features.empty:
        raise ValueError("Training dataset is empty")

    if val_features.empty:
        raise ValueError("Validation dataset is empty")

    device = DEVICE

    train_tensors, scaler = make_tensors(sampled_train_features)
    val_tensors, _ = make_tensors(val_features, scaler)
    n_numeric = train_tensors[0].shape[1]

    train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_tensors), batch_size=BATCH_SIZE, shuffle=False)

    model = NN(n_numeric=n_numeric).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_ndcg = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = {}
    history = {
        "train_loss": [],
        "val_loss": [],
        "auc": [],
        "ndcg@10": [],
        "ndcg@20": [],
    }
    best_epoch = 0

    for epoch in range(EPOCHS):
        avg_train_loss = train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device
                )

        val_loss, metrics = evaluate(
            model,
            val_loader,
            val_features
        )

        if metrics["ndcg@10"] > best_ndcg:
            best_ndcg = metrics["ndcg@10"]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics.copy()
            best_epoch = epoch + 1

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

    return model, best_metrics, best_epoch


def train_final_nn(
    full_train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    epochs: int,
) -> tuple[NN, StandardScaler]:
    device = DEVICE

    final_train_features = pd.concat([full_train_features, val_features], ignore_index=True)
    final_train_features = add_candidate_hardness_scores(final_train_features, user_col="user_id")
    final_train_features = sample_negatives_by_hardness(
        final_train_features,
        negatives_per_positive=10,
        user_col="user_id",
        label_col="label",
        random_state=42
    )

    train_tensors, scaler = make_tensors(final_train_features)
    n_numeric = train_tensors[0].shape[1]
    train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=BATCH_SIZE, shuffle=True)

    model = NN(n_numeric=n_numeric).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []

    for epoch in range(epochs):
        avg_train_loss = train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device
                )

        train_losses.append(avg_train_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train_loss={avg_train_loss:.4f}"
        )

    plt.figure(figsize=(8, 5))
    plt.plot(list(range(1, epochs + 1)), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Final Neural Network Training Loss")
    plt.grid(True)
    plt.show()

    return model, scaler
