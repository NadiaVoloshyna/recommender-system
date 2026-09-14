import numpy as np


# Precision@K: of the top k tracks recommended to the user, how many were actually relevant
def precision_at_k(df, k):
    precisions = []

    for _, group in df.groupby("user_id"):
        top_k = group.sort_values("score", ascending=False).head(k)
        precision = top_k["label"].sum() / len(top_k)
        precisions.append(precision)

    return np.mean(precisions)


# Recall@K: of all the relevant tracks for this user, how many did we manage to put in the top k
def recall_at_k(df, k):
    recalls = []

    for _, group in df.groupby("user_id"):
        total_relevant = group["label"].sum()
        if total_relevant == 0:
            continue

        top_k = group.sort_values("score", ascending=False).head(k)
        relevant_in_top_k = top_k["label"].sum()

        recall = relevant_in_top_k / total_relevant
        recalls.append(recall)

    return np.mean(recalls)


# NDCG@K: where the relevant tracks appear in the ranking
def ndcg_at_k(df, k):
    ndcgs = []

    for _, group in df.groupby("user_id"):
        ranked = group.sort_values("score", ascending=False).head(k)

        relevance = ranked["label"].to_numpy()
        discounts = np.log2(np.arange(2, len(relevance) + 2))
        dcg = np.sum(relevance / discounts)

        # Ideal DCG
        ideal_relevance = np.sort(group["label"].to_numpy())[::-1][:k]
        ideal_discounts = np.log2(np.arange(2, len(ideal_relevance) + 2))
        idcg = np.sum(ideal_relevance / ideal_discounts)
        if idcg == 0:
            continue

        ndcg = dcg / idcg
        ndcgs.append(ndcg)

    return np.mean(ndcgs)


def evaluate_ranker(val_results, ks=(5, 10, 20)):
    results = {}

    for k in ks:
        results[k] = {
            "precision": precision_at_k(val_results, k),
            "recall": recall_at_k(val_results, k),
            "ndcg": ndcg_at_k(val_results, k)
        }

    return results

