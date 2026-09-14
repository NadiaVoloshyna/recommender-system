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


def fit_transform_features(features):
    X_train = features.drop(columns=COLUMNS_TO_DROP)

    scaler = StandardScaler()
    X_train[COLUMNS_TO_SCALE] = scaler.fit_transform(X_train[COLUMNS_TO_SCALE])

    return X_train, scaler


def transform_features(features, scaler):
    X = features.drop(columns=COLUMNS_TO_DROP)
    X[COLUMNS_TO_SCALE] = scaler.transform(X[COLUMNS_TO_SCALE])

    return X


