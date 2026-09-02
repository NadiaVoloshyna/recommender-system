import pandas as pd
import numpy as np


def split_user_history(
    interactions: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split each user's interaction history into training, validation, and test sets.
    Each user is split independently so that the proportions are approximately preserved for every user.
    :param interactions: user-track interaction data. Must contain 'user_id' (pd.DataFrame)
    :param val_size: proportion of each user's interactions assigned to validation (float)
    :param test_size: proportion of each user's interactions assigned to test (float)
    :param random_state: seed for reproducible shuffling (int)
    :return:
        train_interactions: training interactions (pd.DataFrame)
        val_interactions: validation interactions (pd.DataFrame)
        test_interactions: test interactions (pd.DataFrame)
    """
    if not isinstance(interactions, pd.DataFrame):
        raise TypeError("interactions must be a pandas DataFrame")

    if interactions.empty:
        raise ValueError("interactions must not be empty")

    if "user_id" not in interactions.columns:
        raise ValueError("interactions must contain a 'user_id' column")

    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1")

    rng = np.random.RandomState(random_state)

    train_parts = []
    val_parts = []
    test_parts = []

    for _, user_df in interactions.groupby("user_id"):
        # Shuffle user's interactions
        user_df = user_df.sample(frac=1, random_state=rng)
        n = len(user_df)

        # For users with fewer than 3 interactions, keep all interactions in training.
        if n < 3:
            train_parts.append(user_df)
            continue

        n_val = max(1, int(n * val_size))
        n_test = max(1, int(n * test_size))

        # Ensure at least one interaction remains for training
        if n_val + n_test >= n:
            n_test = max(1, int(n * test_size))
            n_val = max(1, n - n_test - 1)

        test = user_df.iloc[:n_test]
        val = user_df.iloc[n_test:n_test + n_val]
        train = user_df.iloc[n_test + n_val:]

        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True)
    )


def split_training_history(
    train_interactions: pd.DataFrame,
    target_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split each user's training interactions into a history set and target set.
    :param train_interactions: training interactions. Must contain 'user_id' (pd.DataFrame)
    :param target_size: proportion of each user's training interactions assigned to the target set (float)
    :param random_state: seed for reproducible shuffling (int)
    :return:
        train_history: interactions used as the user's history (pd.DataFrame)
        train_targets: held-out interactions used as training targets (pd.DataFrame)
    """
    if not isinstance(train_interactions, pd.DataFrame):
        raise TypeError("interactions must be a pandas DataFrame")

    if train_interactions.empty:
        raise ValueError("interactions must not be empty")

    if "user_id" not in train_interactions.columns:
        raise ValueError("interactions must contain a 'user_id' column")

    if not 0 < target_size < 1:
        raise ValueError("target_size must be between 0 and 1")

    rng = np.random.RandomState(random_state)

    history_parts = []
    target_parts = []

    for _, user_df in train_interactions.groupby("user_id"):

        user_df = user_df.sample(frac=1, random_state=rng)
        n = len(user_df)

        n_target = max(1, int(n * target_size))

        # Keep at least one interaction in history
        if n > 1:
            n_target = min(n_target, n - 1)
        else:
            n_target = 0

        targets = user_df.iloc[:n_target]
        history = user_df.iloc[n_target:]

        history_parts.append(history)
        target_parts.append(targets)

    return (
        pd.concat(history_parts, ignore_index=True),
        pd.concat(target_parts, ignore_index=True)
    )


def add_labels(
    feature_df: pd.DataFrame,
    held_out: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add binary labels to the feature dataset.
    Takes the held-out interactions as positive examples and labels matching user-track pairs in the feature dataset
    as positive (1), while all other pairs are labelled as negative (0).
    :param feature_df: one row per user-track pair containing the engineered ranking features (pd.DataFrame)
    :param held_out: held-out test interactions for all users (pd.DataFrame)
    :return: feature dataset containing the engineered ranking features and the binary labels (pd.DataFrame).
    """
    if not isinstance(feature_df, pd.DataFrame):
        raise TypeError("feature_df must be a pandas DataFrame")

    if not isinstance(held_out, pd.DataFrame):
        raise TypeError("held_out must be a pandas DataFrame")

    positive_pairs = held_out[["user_id", "track_id"]].drop_duplicates()

    feature_df = feature_df.copy()

    feature_df["label"] = (
        feature_df.set_index(["user_id", "track_id"]).index
        .isin(positive_pairs.set_index(["user_id", "track_id"]).index)
        .astype("int8")
    )

    return feature_df
