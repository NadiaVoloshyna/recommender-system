import pandas as pd


def sample_negatives(
    feature_df: pd.DataFrame,
    negatives_per_positive: int = 10,
    hard_ratio=0.50,
    medium_ratio=0.30,
    random_ratio=0.20,
    random_state: int = 42,
) -> pd.DataFrame:

