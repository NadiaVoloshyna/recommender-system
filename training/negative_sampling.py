from features.utils import validate_columns
import pandas as pd
import numpy as np

HARDNESS_FEATURES = {
    "source_interaction_strength_log": 0.35,
    "vector_similarity_score": 0.30,
    "artist_similarity_score": 0.15,
    "track_similarity_score": 0.10,
    "candidate_relative_global_popularity": 0.10
}

HARDNESS_BUCKETS = {
    "easy": (0.00, 0.20),
    "medium": (0.20, 0.50),
    "hard": (0.50, 0.80),
    "very_hard": (0.80, 1.00)
}

NEGATIVE_ALLOCATION = {
    "easy": 0.20,
    "medium": 0.30,
    "hard": 0.30,
    "very_hard": 0.20
}


def user_percentile(
    df: pd.DataFrame,
    column: str,
    user_col: str = "user_id",
) -> pd.Series:
    """
    Calculate the percentile rank of a feature within each user.
    Each user's candidates are ranked independently, producing values approximately in [0, 1].
    Higher value = higher value of the original feature for that user.
    :param df: input candidate DataFrame
    :param column: feature column to rank (str)
    :param user_col: user identifier column (str)
    :return: Series containing the per-user percentile ranks.
    """
    feature_series: pd.Series = df[column]

    return (
        feature_series
        .groupby(df[user_col])
        .rank(method="average", pct=True)
    )


def add_candidate_hardness_scores(
    df: pd.DataFrame,
    user_col: str = "user_id",
) -> pd.DataFrame:
    """
    Calculates candidate hardness relative to each user's candidate pool.
    Add per-user percentile features and a combined hardness score. The original feature columns are preserved.
    Added columns: *_rank, hardness_score, hardness_percentile, hardness_bucket
    :param df: input candidate DataFrame
    :param user_col: user identifier column (str)
    :return: DataFrame with hardness features added.
    """
    df = df.copy()

    required_columns = [user_col, *HARDNESS_FEATURES.keys()]
    validate_columns(df, required_columns, "df")

    # Convert each raw feature to a per-user percentile
    for feature in HARDNESS_FEATURES:
        df[f"{feature}_rank"] = user_percentile(
            df=df,
            column=feature,
            user_col=user_col
        )

    # Calculate weighted hardness score
    df["hardness_score"] = 0.0
    for feature, weight in HARDNESS_FEATURES.items():
        df["hardness_score"] += (weight * df[f"{feature}_rank"])

    # Convert combined hardness score to a per-user percentile
    df["hardness_percentile"] = user_percentile(
        df=df,
        column="hardness_score",
        user_col=user_col)

    # Assign hardness buckets
    df["hardness_bucket"] = pd.cut(
        df["hardness_percentile"],
        bins=[0.0, 0.20, 0.50, 0.80, 1.0],
        labels=["easy", "medium", "hard", "very_hard"],
        include_lowest=True)

    return df


def sample_negatives_by_hardness(
    df: pd.DataFrame,
    negatives_per_positive: int = 10,
    label_col: str = "label",
    user_col: str = "user_id",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Keeps all positives and samples negatives according to hardness buckets.
    Default allocation for 10 negatives per positive: easy 20%; medium 30%; hard 30%; very_hard 20%.
    Sampling is performed independently for each user.
    If a hardness bucket does not contain enough negatives, the remaining required negatives are sampled from the
    user's remaining negative pool.
    :param df: input candidate DataFrame
    :param negatives_per_positive: number of negatives to sample per positive (int)
    :param label_col: binary label column (str)
    :param user_col: user identifier column (str)
    :param random_state: random seed (int)
    :return: DataFrame containing all positives and sampled negatives.
    """
    if negatives_per_positive <= 0:
        raise ValueError("negatives_per_positive must be > 0")

    required_columns = [user_col, label_col, "hardness_bucket"]
    validate_columns(df, required_columns, "df")

    rng = np.random.default_rng(random_state)

    sampled_users = []

    for user_id, user_df in df.groupby(user_col, sort=False):
        positives = user_df[user_df[label_col] == 1]
        negatives = user_df[user_df[label_col] == 0]
        if positives.empty:
            continue

        n_positives = len(positives)
        target_n_negatives = (n_positives * negatives_per_positive)

        # Sample from each hardness bucket
        sampled_negative_indices = []

        for bucket, fraction in NEGATIVE_ALLOCATION.items():
            target_n = int(round(target_n_negatives * fraction))
            if target_n == 0:
                continue

            bucket_negatives = negatives[negatives["hardness_bucket"] == bucket]
            if bucket_negatives.empty:
                continue

            n_sample = min(target_n, len(bucket_negatives))

            selected_positions = rng.choice(len(bucket_negatives), size=n_sample, replace=False)
            selected_indices = bucket_negatives.iloc[selected_positions].index
            sampled_negative_indices.extend(selected_indices)

        # Fill any shortfall
        shortfall = (target_n_negatives - len(sampled_negative_indices))
        if shortfall > 0:
            remaining_negatives = negatives.drop(
                index=sampled_negative_indices,
                errors="ignore")

            if not remaining_negatives.empty:
                n_fill = min(shortfall, len(remaining_negatives))
                selected_positions = rng.choice(len(remaining_negatives), size=n_fill, replace=False)
                fill_indices = remaining_negatives.iloc[selected_positions].index
                sampled_negative_indices.extend(fill_indices)

        # Keep all positives + sampled negatives
        sampled_negatives = negatives.loc[sampled_negative_indices]
        sampled_users.append(pd.concat([positives, sampled_negatives]))

    if not sampled_users:
        return df.iloc[0:0].copy()

    return pd.concat(sampled_users, ignore_index=True)
