import numpy as np
import pandas as pd
from features.utils import validate_columns


# Precision@K: of the top k tracks recommended to the user, how many were actually relevant
def precision_at_k(df: pd.DataFrame, k: int) -> float:
    """
    Calculates Precision@K for a recommendation model.
    :param df: validation results containing one row per user-track pair, with the user's ID, track ID,
    actual relevance label, and predicted score from the model (pd.DataFrame)
    :param k: number of top-ranked tracks to consider when calculating Precision@K for each user (int)
    :return: the average Precision@K across all users (float)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if df.empty:
        raise ValueError("df must not be empty")

    if not isinstance(k, int):
        raise TypeError("k must be an integer")

    if k <= 0:
        raise ValueError("k must be greater than 0")

    required_columns = ["user_id", "track_id", "label", "score"]
    validate_columns(df, required_columns, "df")

    precisions = []

    for _, group in df.groupby("user_id"):
        top_k = group.sort_values("score", ascending=False).head(k)
        precision = top_k["label"].sum() / len(top_k)
        precisions.append(precision)

    return float(np.mean(precisions))


# Recall@K: of all the relevant tracks for this user, how many did we manage to put in the top k
def recall_at_k(df: pd.DataFrame, k: int) -> float:
    """
    Calculates Recall@K for a recommendation model.
    :param df: validation results containing one row per user-track pair, with the user's ID,
    track ID, actual relevance label, and predicted score from the model (pd.DataFrame)
    :param k: number of top-ranked tracks to consider when calculating Recall@K for each user (int)
    :return: the average Recall@K across users with at least one relevant track (float)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if df.empty:
        raise ValueError("df must not be empty")

    if not isinstance(k, int):
        raise TypeError("k must be an integer")

    if k <= 0:
        raise ValueError("k must be greater than 0")

    required_columns = ["user_id", "track_id", "label", "score"]
    validate_columns(df, required_columns, "df")

    recalls = []

    for _, group in df.groupby("user_id"):
        total_relevant = group["label"].sum()
        if total_relevant == 0:
            continue

        top_k = group.sort_values("score", ascending=False).head(k)
        relevant_in_top_k = top_k["label"].sum()

        recall = relevant_in_top_k / total_relevant
        recalls.append(recall)

    return float(np.mean(recalls))


# NDCG@K: where the relevant tracks appear in the ranking
def ndcg_at_k(df: pd.DataFrame, k: int) -> float:
    """
    Calculates NDCG@K for a recommendation model.
    :param df: validation results containing one row per user-track pair, with the user's ID, track ID,
    actual relevance label, and predicted score from the model (pd.DataFrame)
    :param k: number of top-ranked tracks to consider when calculating NDCG@K for each user (int)
    :return: the average NDCG@K across users with at least one relevant track (float)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if df.empty:
        raise ValueError("df must not be empty")

    if not isinstance(k, int):
        raise TypeError("k must be an integer")

    if k <= 0:
        raise ValueError("k must be greater than 0")

    required_columns = ["user_id", "track_id", "label", "score"]
    validate_columns(df, required_columns, "df")

    ndcgs = []

    for _, group in df.groupby("user_id"):
        ranked = group.sort_values("score", ascending=False).head(k)
        relevance = ranked["label"].to_numpy()
        discounts = np.log2(np.arange(2, len(relevance) + 2))
        dcg = np.sum(relevance / discounts)

        ideal_relevance = np.sort(group["label"].to_numpy())[::-1][:k]
        ideal_discounts = np.log2(np.arange(2, len(ideal_relevance) + 2))
        idcg = np.sum(ideal_relevance / ideal_discounts)
        if idcg == 0:
            continue

        ndcg = dcg / idcg
        ndcgs.append(ndcg)

    return float(np.mean(ndcgs))


def evaluate_ranker(val_results, ks=(10, 20)):
    results = {}

    for k in ks:
        results[k] = {
            "precision": precision_at_k(val_results, k),
            "recall": recall_at_k(val_results, k),
            "ndcg": ndcg_at_k(val_results, k)
        }

    return results

