import pandas as pd
from sklearn.preprocessing import StandardScaler

COLUMNS_TO_DROP = [
    "user_id",
    "track_id",
    "source_interaction_strength",
    "global_popularity",
    "label"
]

COLUMNS_TO_SCALE = [
    "source_interaction_strength_log",
    "n_sources",
    "track_similarity_score",
    "artist_similarity_score",
    "vector_similarity_score",
    "max_similarity",
    "mean_similarity_available",
    "mean_similarity_all",
    "track_interaction_signal",
    "artist_interaction_signal",
    "vector_interaction_signal",
    "global_popularity_log",
    "candidate_relative_global_popularity"
]


def fit_transform_features(features: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Prepares training features for a machine-learning model by removing unwanted columns and
    standardizing selected numerical columns.
    :param features: one row per user-track pair containing the engineered ranking features (pd.DataFrame)
    :return:
        X_train (pd.DataFrame): training features with unwanted columns removed
        and selected numerical columns standardized
        scaler (StandardScaler): fitted scaler used to standardize the selected numerical columns
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")

    if features.empty:
        raise ValueError("features must not be empty")

    X_train = features.drop(columns=COLUMNS_TO_DROP)

    scaler = StandardScaler()
    X_train[COLUMNS_TO_SCALE] = scaler.fit_transform(X_train[COLUMNS_TO_SCALE])

    return X_train, scaler


def transform_features(features: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """
    Prepares new data, such as validation or test data, using the same preprocessing
    that was applied to the training data.
    :param features: one row per user-track pair containing the engineered ranking features (pd.DataFrame)
    :param scaler: fitted scaler used to standardize the selected numerical columns (StandardScaler)
    :return: X: features with unwanted columns removed and selected numerical columns
    standardized using the fitted scaler (pd.DataFrame)
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")

    if features.empty:
        raise ValueError("features must not be empty")

    if not isinstance(scaler, StandardScaler):
        raise TypeError("scaler must be a StandardScaler")

    X = features.drop(columns=COLUMNS_TO_DROP)
    X[COLUMNS_TO_SCALE] = scaler.transform(X[COLUMNS_TO_SCALE])

    return X



