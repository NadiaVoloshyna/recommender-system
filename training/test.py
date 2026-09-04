from training.negative_sampling import \
    add_candidate_hardness_scores, \
    HARDNESS_FEATURES, \
    sample_negatives_by_hardness, \
    NEGATIVE_ALLOCATION
import pandas as pd
import numpy as np


def make_candidate_df():
    df = pd.DataFrame({
        "user_id": ["A", "A", "A", "B", "B", "B"],
        "source_interaction_strength_log": [
            1.0, 2.0, 3.0,
            10.0, 20.0, 30.0,
        ],
        "vector_similarity_score": [1, 2, 3, 1, 2, 3],
        "track_similarity_score": [1, 2, 3, 1, 2, 3],
        "artist_similarity_score": [1, 2, 3, 1, 2, 3],
        "candidate_relative_global_popularity": [1, 2, 3, 1, 2, 3],
    })

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
        for feature, weight in HARDNESS_FEATURES.items()
    )

    pd.testing.assert_series_equal(
        result["hardness_score"],
        expected,
        check_names=False,
    )


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
                    "hardness_bucket": None,
                }
            )
            row_id += 1

        for i in range(negatives_per_user):
            rows.append(
                {
                    "row_id": row_id,
                    "user_id": user_id,
                    "track_id": f"{user_id}_negative_{i}",
                    "label": 0,
                    "hardness_bucket": buckets[i % len(buckets)],
                }
            )
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
                "hardness_bucket": None,
            }
        )
        row_id += 1

    for bucket in BUCKETS:
        for i in range(negatives_per_bucket):
            rows.append(
                {
                    "row_id": row_id,
                    "user_id": "user_1",
                    "track_id": f"{bucket}_{i}",
                    "label": 0,
                    "hardness_bucket": bucket,
                }
            )
            row_id += 1

    return pd.DataFrame(rows).set_index("row_id")


# sample_negatives_by_hardness() keeps all positives
def test_sample_negatives_by_hardness_keeps_all_positives():
    df = make_test_df(
        positives_per_user={"user_1": 3},
        negatives_per_user=100
    )

    expected_positive_ids = set(df.loc[df["label"] == 1, "track_id"])

    result = sample_negatives_by_hardness(
        df,
        negatives_per_positive=10,
        random_state=42
    )

    actual_positive_ids = set(result.loc[result["label"] == 1, "track_id"])

    assert actual_positive_ids == expected_positive_ids


# sample_negatives_by_hardness() samples correct number of negatives
def test_sample_negatives_by_hardness_samples_correct_number_of_negatives():
    df = make_test_df(
        positives_per_user={"user_1": 4},
        negatives_per_user=100,
    )

    result = sample_negatives_by_hardness(
        df,
        negatives_per_positive=10,
        random_state=42,
    )

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
        negatives_per_user=100,
    )

    result = sample_negatives_by_hardness(
        df,
        negatives_per_positive=10,
        random_state=42,
    )

    for user_id, user_result in result.groupby("user_id"):
        n_positives = (user_result["label"] == 1).sum()

        n_negatives = (user_result["label"] == 0).sum()

        assert n_negatives == (n_positives * 10)


# sample_negatives_by_hardness(): correct hardness allocation
def test_sample_negatives_by_hardness_hardness_allocation():
    """
    With: 1 positive, 10 negatives requested
    and allocation:
        easy       = 20% -> 2
        medium     = 30% -> 3
        hard       = 30% -> 3
        very_hard  = 20% -> 2
    """
    df = make_balanced_bucket_df(
        n_positives=1,
        negatives_per_bucket=10,
    )

    result = sample_negatives_by_hardness(
        df,
        negatives_per_positive=10,
        random_state=42,
    )

    sampled_negatives = result[result["label"] == 0]

    counts = (
        sampled_negatives["hardness_bucket"]
        .value_counts()
        .to_dict()
    )

    assert counts["easy"] == 2
    assert counts["medium"] == 3
    assert counts["hard"] == 3
    assert counts["very_hard"] == 2


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
            "hard",
        ],
    )

    result = sample_negatives_by_hardness(
        df,
        negatives_per_positive=10,
        random_state=42,
    )

    n_negatives = (result["label"] == 0).sum()

    assert n_negatives == 10



