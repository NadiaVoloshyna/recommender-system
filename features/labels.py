import pandas as pd


def split_user_history(
    interactions: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split each user's interaction history randomly into training and test sets.
    Each user has at least one interaction in the test set and, when possible,
    at least one interaction in the training set.
    :param interactions: user-track interaction data (pd.DataFrame)
    :param test_size: proportion of each user's interactions assigned to the test set (float)
    :param random_state: seed for reproducible random shuffling (int)
    :return: a tuple containing the combined training and test DataFrames.
    """
    if not isinstance(interactions, pd.DataFrame):
        raise TypeError("interactions must be a pandas DataFrame")

    if interactions.empty:
        raise ValueError("interactions must not be empty")

    if "user_id" not in interactions.columns:
        raise ValueError("interactions must contain a 'user_id' column")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    train_parts = []
    test_parts = []

    for _, user_df in interactions.groupby("user_id"):
        user_df = user_df.sample(frac=1, random_state=random_state)

        n_test = max(1, int(len(user_df) * test_size))
        if len(user_df) > 1:
            n_test = min(n_test, len(user_df) - 1)

        test = user_df.iloc[:n_test]
        train = user_df.iloc[n_test:]

        train_parts.append(train)
        test_parts.append(test)

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True)
    )

