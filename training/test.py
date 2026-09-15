from training.negative_sampling import add_candidate_hardness_scores, HARDNESS_FEATURES, sample_negatives_by_hardness
from training.preprocessing import fit_transform_features, transform_features, COLUMNS_TO_DROP, COLUMNS_TO_SCALE
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pytest


def make_candidate_df():
    df = pd.DataFrame({
        "user_id": ["A", "A", "A", "B", "B", "B"],
        "source_interaction_strength_log": [
            1.0, 2.0, 3.0,
            10.0, 20.0, 30.0],
        "vector_similarity_score": [1, 2, 3, 1, 2, 3],
        "track_similarity_score": [1, 2, 3, 1, 2, 3],
        "artist_similarity_score": [1, 2, 3, 1, 2, 3],
        "candidate_relative_global_popularity": [1, 2, 3, 1, 2, 3]})

    return df


def make_full_candidate_df():
    df = pd.DataFrame({
        "user_id": ["A", "A", "A", "B", "B", "B"],
        "track_id": ["T1", "T2", "T3", "T1", "T2", "T3"],
        "source_interaction_strength": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        "source_interaction_strength_log": [
            1.0, 2.0, 3.0,
            10.0, 20.0, 30.0],
        "n_sources": [1, 2, 3, 1, 2, 3],
        "track_similarity_score": [1, 2, 3, 1, 2, 3],
        "artist_similarity_score": [1, 2, 3, 1, 2, 3],
        "vector_similarity_score": [1, 2, 3, 1, 2, 3],
        "max_similarity": [1, 2, 3, 1, 2, 3],
        "mean_similarity_available": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "mean_similarity_all": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "track_interaction_signal": [1.0, 4.0, 9.0, 10.0, 40.0, 90.0],
        "artist_interaction_signal": [1.0, 4.0, 9.0, 10.0, 40.0, 90.0],
        "vector_interaction_signal": [1.0, 4.0, 9.0, 10.0, 40.0, 90.0],
        "global_popularity": [1, 2, 3, 1, 2, 3],
        "global_popularity_log": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "candidate_relative_global_popularity": [1, 2, 3, 1, 2, 3],
        "global_popularity_missing": [0, 0, 0, 0, 0, 0],
        "track_similarity_available": [1, 1, 1, 1, 1, 1],
        "artist_similarity_available": [1, 1, 1, 1, 1, 1],
        "vector_similarity_available": [1, 1, 1, 1, 1, 1],
        "label": [0, 1, 0, 1, 0, 1]})

    return df


# add_candidate_hardness_scores() adds all expected columns
def test_add_candidate_hardness_scores_adds_expected_columns():
    df = make_candidate_df()

    result = add_candidate_hardness_scores(df)

    for feature in HARDNESS_FEATURES:
        assert f"{feature}_rank" in result.columns

    assert "hardness_score" in result.columns
    assert "hardness_percentile" in result.columns
    assert "hardness_bucket" in result.columns


# add_candidate_hardness_scores() does not mutate input
def test_add_candidate_hardness_scores_does_not_modify_input():
    df = make_candidate_df()
    original = df.copy(deep=True)

    add_candidate_hardness_scores(df)

    pd.testing.assert_frame_equal(df, original)


# add_candidate_hardness_scores() ranks independently per user
def test_add_candidate_hardness_scores_percentile_ranks_are_calculated_per_user():
    df = make_candidate_df()

    result = add_candidate_hardness_scores(df)

    expected = [1 / 3, 2 / 3, 1.0] * 2

    actual = result["source_interaction_strength_log_rank"]
    np.testing.assert_allclose(actual, expected)


# add_candidate_hardness_scores() Calculates weighted hardness correctly
def test_add_candidate_hardness_scores_uses_configured_weights():
    df = make_candidate_df()

    result = add_candidate_hardness_scores(df)

    expected = sum(
        weight * result[f"{feature}_rank"]
        for feature, weight in HARDNESS_FEATURES.items())

    pd.testing.assert_series_equal(
        result["hardness_score"],
        expected,
        check_names=False)


# add_candidate_hardness_scores() Assigns valid hardness buckets
def test_add_candidate_hardness_scores_bucket_values_are_valid():
    df = make_candidate_df()
    result = add_candidate_hardness_scores(df)
    valid_buckets = {"easy", "medium", "hard", "very_hard"}

    assert set(
        result["hardness_bucket"].dropna().astype(str)
    ).issubset(valid_buckets)


BUCKETS = ["easy", "medium", "hard", "very_hard"]


def make_test_df(
    positives_per_user: dict[str, int],
    negatives_per_user: int,
    buckets: list[str] | None = None,
) -> pd.DataFrame:
    """
    Creates a synthetic candidate DataFrame. Each user receives: the requested number of positives,
    the requested number of negatives. Negative candidates are distributed across hardness buckets
    in round-robin fashion unless `buckets` is supplied.
    """
    if buckets is None:
        buckets = BUCKETS

    rows = []
    row_id = 1000

    for user_id, n_positives in positives_per_user.items():
        for i in range(n_positives):
            rows.append(
                {
                    "row_id": row_id,
                    "user_id": user_id,
                    "track_id": f"{user_id}_positive_{i}",
                    "label": 1,
                    "hardness_bucket": None})
            row_id += 1

        for i in range(negatives_per_user):
            rows.append(
                {
                    "row_id": row_id,
                    "user_id": user_id,
                    "track_id": f"{user_id}_negative_{i}",
                    "label": 0,
                    "hardness_bucket": buckets[i % len(buckets)]})
            row_id += 1

    return pd.DataFrame(rows).set_index("row_id")


def make_balanced_bucket_df(
    n_positives: int = 1,
    negatives_per_bucket: int = 10,
) -> pd.DataFrame:
    """
    Creates one user's candidate pool with an equal number of negatives in every hardness bucket.
    """
    rows = []
    row_id = 1000

    for i in range(n_positives):
        rows.append(
            {
                "row_id": row_id,
                "user_id": "user_1",
                "track_id": f"positive_{i}",
                "label": 1,
                "hardness_bucket": None})
        row_id += 1

    for bucket in BUCKETS:
        for i in range(negatives_per_bucket):
            rows.append(
                {
                    "row_id": row_id,
                    "user_id": "user_1",
                    "track_id": f"{bucket}_{i}",
                    "label": 0,
                    "hardness_bucket": bucket})
            row_id += 1

    return pd.DataFrame(rows).set_index("row_id")


# sample_negatives_by_hardness() keeps all positives
def test_sample_negatives_by_hardness_keeps_all_positives():
    df = make_test_df(
        positives_per_user={"user_1": 3},
        negatives_per_user=100)
    expected_positive_ids = set(df.loc[df["label"] == 1, "track_id"])

    result = sample_negatives_by_hardness(
        df,
        negatives_per_positive=10,
        random_state=42)

    actual_positive_ids = set(result.loc[result["label"] == 1, "track_id"])

    assert actual_positive_ids == expected_positive_ids


# sample_negatives_by_hardness() samples correct number of negatives
def test_sample_negatives_by_hardness_samples_correct_number_of_negatives():
    df = make_test_df(
        positives_per_user={"user_1": 4},
        negatives_per_user=100)

    result = sample_negatives_by_hardness(
        df,
        negatives_per_positive=10,
        random_state=42)

    n_positives = (result["label"] == 1).sum()
    n_negatives = (result["label"] == 0).sum()

    assert n_positives == 4
    assert n_negatives == 40


# sample_negatives_by_hardness(): sampling is independent per user
def test_sample_negatives_by_hardness_sampling_is_independent_per_user():
    df = make_test_df(
        positives_per_user={
            "user_1": 2,
            "user_2": 5,
        },
        negatives_per_user=100)

    result = sample_negatives_by_hardness(
        df,
        negatives_per_positive=10,
        random_state=42)

    for user_id, user_result in result.groupby("user_id"):
        n_positives = (user_result["label"] == 1).sum()
        n_negatives = (user_result["label"] == 0).sum()

        assert n_negatives == (n_positives * 10)


# sample_negatives_by_hardness() handles missing bucket. shortfall
def test_missing_hardness_bucket_fills_shortfall():
    """
    There are no very_hard negatives. The function should still produce the requested
    number of negatives by sampling from the remaining pool.
    """
    df = make_test_df(
        positives_per_user={"user_1": 1},
        negatives_per_user=40,
        buckets=[
            "easy",
            "medium",
            "hard"])
    result = sample_negatives_by_hardness(
        df,
        negatives_per_positive=10,
        random_state=42)

    n_negatives = (result["label"] == 0).sum()

    assert n_negatives == 10


# fit_transform_features() returns correct types
def test_fit_transform_features_returns_correct_types():
    features = make_full_candidate_df()

    X_train, scaler = fit_transform_features(features)

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(scaler, StandardScaler)


# fit_transform_features() removes unwanted columns
def test_fit_transform_features_removes_unwanted_columns():
    features = make_full_candidate_df()

    X_train, scaler = fit_transform_features(features)

    for column in COLUMNS_TO_DROP:
        assert column not in X_train.columns


# fit_transform_features() standardizes selected columns
def test_fit_transform_features_standardizes_selected_columns():
    features = make_full_candidate_df()

    X_train, scaler = fit_transform_features(features)

    for column in COLUMNS_TO_SCALE:
        assert X_train[column].mean() == pytest.approx(0)
        assert X_train[column].std(ddof=0) == pytest.approx(1)


# transform_features() removes unwanted columns
def test_transform_features_removes_unwanted_columns():
    features = make_full_candidate_df()
    scaler = StandardScaler()
    scaler.fit(features[COLUMNS_TO_SCALE])

    X = transform_features(features, scaler)

    for column in COLUMNS_TO_DROP:
        assert column not in X.columns


# transform_features()  uses fitted scaler
def test_transform_features_uses_fitted_scaler():
    features = make_full_candidate_df()
    scaler = StandardScaler()
    scaler.fit(features[COLUMNS_TO_SCALE])

    X = transform_features(features, scaler)

    expected = scaler.transform(features[COLUMNS_TO_SCALE])

    pd.testing.assert_frame_equal(
        X[COLUMNS_TO_SCALE].reset_index(drop=True),
        pd.DataFrame(expected, columns=COLUMNS_TO_SCALE))






