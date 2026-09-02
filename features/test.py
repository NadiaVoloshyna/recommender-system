import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from features.interaction_builder import build_interaction_dataframe
from features.embedding_builder import build_track_embeddings
from features.candidate_features import build_candidate_features
from features.labels import split_user_history, add_labels


# build_interaction_dataframe() builds interactions correctly
def test_build_interaction_dataframe_builds_interactions():
    recent_tracks_df = pd.DataFrame({
        "user_id": [1],
        "track_name": ["Song A"],
        "artist_name": ["Artist A"]
    })

    top_tracks_df = pd.DataFrame({
        "user_id": [1],
        "track_name": ["Song B"],
        "artist_name": ["Artist B"],
        "playcount": [9]
    })

    tracks_df = pd.DataFrame({
        "track_id": [10, 20],
        "track_name": ["Song A", "Song B"],
        "artist_name": ["Artist A", "Artist B"]
    })

    result = build_interaction_dataframe(recent_tracks_df, top_tracks_df, tracks_df)

    expected = pd.DataFrame({
        "user_id": [1, 1],
        "track_id": [10, 20],
        "interaction_strength": [1.0, np.log1p(9)]
    })

    pdt.assert_frame_equal(
        result.sort_values("track_id").reset_index(drop=True),
        expected
    )


# build_interaction_dataframe() aggregates duplicate interactions
def test_build_interaction_dataframe_sums_duplicate_interactions():
    recent_tracks_df = pd.DataFrame({
        "user_id": [1],
        "track_name": ["Song A"],
        "artist_name": ["Artist A"]
    })

    top_tracks_df = pd.DataFrame({
        "user_id": [1],
        "track_name": ["Song A"],
        "artist_name": ["Artist A"],
        "playcount": [9]
    })

    tracks_df = pd.DataFrame({
        "track_id": [10],
        "track_name": ["Song A"],
        "artist_name": ["Artist A"]
    })

    result = build_interaction_dataframe(recent_tracks_df, top_tracks_df, tracks_df)

    assert len(result) == 1
    assert result.loc[0, "interaction_strength"] == pytest.approx(
        1 + np.log1p(9)
    )


# build_interaction_dataframe() removes unmatched tracks
def test_build_interaction_dataframe_removes_unmatched_tracks():
    recent_tracks_df = pd.DataFrame({
        "user_id": [1],
        "track_name": ["Unknown"],
        "artist_name": ["Nobody"]
    })

    top_tracks_df = pd.DataFrame(columns=[
        "user_id",
        "track_name",
        "artist_name",
        "playcount"
    ])

    tracks_df = pd.DataFrame({
        "track_id": [1],
        "track_name": ["Song A"],
        "artist_name": ["Artist A"]
    })

    with pytest.raises(ValueError, match="Interaction dataframe is empty"):
        build_interaction_dataframe(recent_tracks_df, top_tracks_df, tracks_df)


# build_interaction_dataframe() raises an error for empty inputs
def test_empty_inputs_raise_error():
    recent_tracks_df = pd.DataFrame(columns=[
        "user_id",
        "track_name",
        "artist_name"
    ])

    top_tracks_df = pd.DataFrame(columns=[
        "user_id",
        "track_name",
        "artist_name",
        "playcount"
    ])

    tracks_df = pd.DataFrame(columns=[
        "track_id",
        "track_name",
        "artist_name"
    ])

    with pytest.raises(ValueError, match="No interaction data available"):
        build_interaction_dataframe(recent_tracks_df, top_tracks_df, tracks_df)


# build_interaction_dataframe() raises an error for missing required columns
def test_missing_columns_raise_error():
    recent_tracks_df = pd.DataFrame({
        "user_id": [1]
    })

    top_tracks_df = pd.DataFrame(columns=[
        "user_id",
        "track_name",
        "artist_name",
        "playcount"
    ])

    tracks_df = pd.DataFrame(columns=[
        "track_id",
        "track_name",
        "artist_name"
    ])

    with pytest.raises(ValueError):
        build_interaction_dataframe(recent_tracks_df, top_tracks_df, tracks_df)


class FakeSentenceTransformer:
    """
    Mock model replacing SentenceTransformer during tests.
    Returns a fixed-size embedding vector.
    """
    def encode(self, text):
        return np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0
            ]
        )


@pytest.fixture
def mock_sentence_transformer(monkeypatch):
    """
    Replace the real SentenceTransformer with the fake model.
    """
    monkeypatch.setattr(
        "features.embedding_builder.SentenceTransformer",
        lambda model_name: FakeSentenceTransformer()
    )


# build_track_embeddings() generates embedding
def test_build_track_embeddings_returns_embeddings(mock_sentence_transformer):
    tracks_df = pd.DataFrame(
        {
            "track_id": [1, 2],
            "artist_name": ["Coldplay", "Adele"],
            "track_name": ["Yellow", "Hello"]
        }
    )

    result = build_track_embeddings(tracks_df)

    assert isinstance(result, dict)
    assert len(result) == 2
    assert 1 in result
    assert 2 in result
    assert isinstance(result[1], np.ndarray)


# build_track_embeddings() produces correct embedding shape
def test_build_track_embeddings_produces_correct_dimension(mock_sentence_transformer):
    tracks_df = pd.DataFrame(
        {
            "track_id": [1],
            "artist_name": ["Coldplay"],
            "track_name": ["Yellow"]
        }
    )

    result = build_track_embeddings(tracks_df)

    embedding = result[1]

    assert embedding.shape == (4,)


# build_track_embeddings() normalizes embeddings
def test_build_track_embeddings_normalizes_embeddings(mock_sentence_transformer):
    tracks_df = pd.DataFrame(
        {
            "track_id": [1],
            "artist_name": ["Coldplay"],
            "track_name": ["Yellow"]
        }
    )

    result = build_track_embeddings(tracks_df)

    embedding = result[1]

    vector_length = np.linalg.norm(embedding)

    assert vector_length == pytest.approx(1.0)


# build_track_embeddings() raises error for missing required columns
def test_build_track_embeddings_raises_error(mock_sentence_transformer):
    tracks_df = pd.DataFrame(
        {
            "track_id": [1],
            "track_name": ["Yellow"]
            # artist_name missing
        }
    )

    with pytest.raises(ValueError):
        build_track_embeddings(tracks_df)


# build_track_embeddings() handles missing artist name
def test_build_track_embeddings_handles_missing_artist_name(mock_sentence_transformer):
    tracks_df = pd.DataFrame(
        {
            "track_id": [1],
            "artist_name": [np.nan],
            "track_name": ["Yellow"]
        }
    )

    result = build_track_embeddings(tracks_df)

    assert 1 in result
    assert isinstance(result[1], np.ndarray)


# build_track_embeddings() handles missing track name
def test_build_track_embeddings_handles_missing_track_name(mock_sentence_transformer):
    tracks_df = pd.DataFrame(
        {
            "track_id": [1],
            "artist_name": ["Coldplay"],
            "track_name": [np.nan]
        }
    )

    result = build_track_embeddings(tracks_df)

    assert 1 in result
    assert isinstance(result[1], np.ndarray)


def make_test_data():
    candidates = pd.DataFrame({
        "user_id": ["u1", "u1", "u1", "u1", "u2"],
        "track_id": ["t1", "t1", "t2", "t3", "t4"],
        "interaction_strength": [2.0, 3.0, 4.0, 1.0, 2.0],
        "track_similarity_score": [0.8, np.nan, 0.5, 0.0, 0.3],
        "artist_similarity_score": [0.0, 0.7, np.nan, 0.0, 0.0],
        "vector_similarity_score": [0.0, 0.0, 0.0, 0.6, 0.0],
        "source": ["track", "artist", "track", "vector", "track"],
    })

    interaction_df = pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u3"],
        "track_id": ["t1", "t2", "t2", "t3"],
        "interaction_strength": [10.0, 20.0, 5.0, 2.0],
    })

    return candidates, interaction_df


# build_candidate_features(): one row per user-track
def test_build_candidate_features_one_row_per_user_track():
    candidates, interaction_df = make_test_data()

    result = build_candidate_features(candidates, interaction_df)

    assert not result.duplicated(["user_id", "track_id"]).any()


# build_candidate_features() consolidates multiple sources correctly
def test_build_candidate_features_aggregates_multiple_sources():
    candidates, interaction_df = make_test_data()

    result = build_candidate_features(candidates, interaction_df)

    row = result[(result["user_id"] == "u1") & (result["track_id"] == "t1")].iloc[0]

    assert row["n_sources"] == 2
    assert row["source_interaction_strength"] == 3.0
    assert row["track_similarity_score"] == 0.8
    assert row["artist_similarity_score"] == 0.7


# build_candidate_features() makes missing similarity values become zero
def test_build_candidate_features_missing_similarity_values_are_zero():
    candidates, interaction_df = make_test_data()

    result = build_candidate_features(candidates, interaction_df)

    assert result[
        ["track_similarity_score",
         "artist_similarity_score",
         "vector_similarity_score"]
    ].isna().sum().sum() == 0


# build_candidate_features(): log interaction strength
def test_build_candidate_features_interaction_log():
    candidates, interaction_df = make_test_data()

    result = build_candidate_features(candidates, interaction_df)

    row = result[(result["user_id"] == "u1") & (result["track_id"] == "t1")].iloc[0]

    expected = np.log1p(3.0)

    assert row["source_interaction_strength_log"] == pytest.approx(expected)


# build_candidate_features(): similarity availability flags
def test_build_candidate_features_similarity_availability_flags():
    candidates, interaction_df = make_test_data()

    result = build_candidate_features(candidates, interaction_df)

    row = result[(result["user_id"] == "u1") & (result["track_id"] == "t1")].iloc[0]

    assert row["track_similarity_available"] == 1
    assert row["artist_similarity_available"] == 1
    assert row["vector_similarity_available"] == 0


# build_candidate_features(): global popularity
def test_build_candidate_features_global_popularity():
    candidates, interaction_df = make_test_data()

    result = build_candidate_features(candidates, interaction_df)

    row = result[(result["user_id"] == "u1") & (result["track_id"] == "t2")].iloc[0]

    assert row["global_popularity"] == 25.0
    assert row["global_popularity_log"] == pytest.approx(np.log1p(25.0))


# build_candidate_features(): missing global popularity
def test_build_candidate_features_missing_global_popularity():
    candidates, interaction_df = make_test_data()

    result = build_candidate_features(candidates, interaction_df)

    row = result[(result["user_id"] == "u2") & (result["track_id"] == "t4")].iloc[0]

    assert row["global_popularity"] == 0
    assert row["global_popularity_missing"] == 1
    assert row["global_popularity_log"] == 0


# build_candidate_features(): candidate relative global popularity
def test_build_candidate_features_relative_global_popularity():
    candidates, interaction_df = make_test_data()

    result = build_candidate_features(candidates, interaction_df)

    row = result[(result["user_id"] == "u1") & (result["track_id"] == "t2")].iloc[0]

    assert row["candidate_relative_global_popularity"] == pytest.approx(1.0)


# build_candidate_features(): interaction × similarity signals
def test_build_candidate_features_interaction_similarity_signal():
    candidates, interaction_df = make_test_data()

    result = build_candidate_features(candidates, interaction_df)

    row = result[(result["user_id"] == "u1") & (result["track_id"] == "t1")].iloc[0]

    expected = np.log1p(3.0) * 0.8

    assert row["track_interaction_signal"] == pytest.approx(expected)


def test_build_candidate_features_zero_similarity_produces_zero_signal():
    candidates, interaction_df = make_test_data()

    result = build_candidate_features(candidates, interaction_df)

    row = result[(result["user_id"] == "u1") & (result["track_id"] == "t1")].iloc[0]

    assert row["vector_interaction_signal"] == 0


# build_candidate_features(): output order
def test_build_candidate_features_output_columns():
    candidates, interaction_df = make_test_data()

    result = build_candidate_features(candidates, interaction_df)

    expected_columns = [
        "user_id",
        "track_id",

        "source_interaction_strength",
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

        "global_popularity",
        "global_popularity_log",
        "candidate_relative_global_popularity",
        "global_popularity_missing",

        "track_similarity_available",
        "artist_similarity_available",
        "vector_similarity_available",
    ]

    assert result.columns.tolist() == expected_columns


# split_user_history() returns three dataframes
def test_split_user_history_returns_three_dataframes():
    interactions = pd.DataFrame({
        "user_id": [1, 1, 1, 1, 2, 2, 2, 2],
        "track_id": ["A", "B", "C", "D", "E", "F", "G", "H"]
    })

    train, val, test = split_user_history(interactions)

    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)


# split_user_history(): each user has validation and test interactions
def test_split_user_history_each_user_has_val_and_test_interaction():
    interactions = pd.DataFrame({
        "user_id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        "track_id": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    })

    train, val, test = split_user_history(
        interactions,
        val_size=0.1,
        test_size=0.1,
        random_state=42,
    )

    assert set(val["user_id"]) == {1, 2, 3}
    assert set(test["user_id"]) == {1, 2, 3}


# split_user_history(): each user keeps training data when possible
def test_split_user_history_each_user_keeps_training_data():
    interactions = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "track_id": ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    })

    train, val, test = split_user_history(
        interactions,
        val_size=0.1,
        test_size=0.1,
        random_state=42,
    )

    assert set(train["user_id"]) == {1, 2, 3}


# split_user_history(): data is not lost or duplicated
def test_split_user_history_data_is_not_lost_or_duplicated():
    interactions = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 2],
        "track_id": ["A", "B", "C", "D", "E", "F"]
    })

    train, val, test = split_user_history(
        interactions,
        val_size=0.1,
        test_size=0.1,
        random_state=42,
    )

    # No rows are lost
    assert len(train) + len(val) + len(test) == len(interactions)

    # No interaction appears in more than one split
    train_rows = set(map(tuple, train.to_numpy()))
    val_rows = set(map(tuple, val.to_numpy()))
    test_rows = set(map(tuple, test.to_numpy()))

    assert train_rows.isdisjoint(val_rows)
    assert train_rows.isdisjoint(test_rows)
    assert val_rows.isdisjoint(test_rows)


# split_user_history(): split is reproducible
def test_split_user_history_split_is_reproducible():
    interactions = pd.DataFrame({
        "user_id": [1, 1, 1, 1, 2, 2, 2, 2],
        "track_id": ["A", "B", "C", "D", "E", "F", "G", "H"]
    })

    train1, val1, test1 = split_user_history(
        interactions,
        val_size=0.1,
        test_size=0.1,
        random_state=42,
    )

    train2, val2, test2 = split_user_history(
        interactions,
        val_size=0.1,
        test_size=0.1,
        random_state=42,
    )

    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(val1, val2)
    pd.testing.assert_frame_equal(test1, test2)


# split_user_history(): different random seeds can produce different splits
def test_split_user_history_different_seed_changes_split():
    interactions = pd.DataFrame({
        "user_id": [1] * 20,
        "track_id": [f"track_{i}" for i in range(20)]
    })

    train1, val1, test1 = split_user_history(
        interactions,
        val_size=0.1,
        test_size=0.1,
        random_state=42,
    )

    train2, val2, test2 = split_user_history(
        interactions,
        val_size=0.1,
        test_size=0.1,
        random_state=123,
    )

    # At least one split should differ
    assert not (
        train1.equals(train2)
        and val1.equals(val2)
        and test1.equals(test2)
    )


# add_labels() labels matching pairs positively
def test_add_labels_matching_pairs_are_labelled_positive():
    feature_df = pd.DataFrame({
        "user_id": [1, 1, 2],
        "track_id": ["A", "B", "C"],
        "score": [0.5, 0.7, 0.2]
    })

    held_out = pd.DataFrame({
        "user_id": [1, 2],
        "track_id": ["A", "C"]
    })

    result = add_labels(feature_df, held_out)

    assert result["label"].tolist() == [1, 0, 1]


# add_labels() labels non-matching pairs negatively
def test_add_labels_non_matching_pairs_are_labelled_negative():
    feature_df = pd.DataFrame({
        "user_id": [1, 1],
        "track_id": ["A", "B"]
    })

    held_out = pd.DataFrame({
        "user_id": [1],
        "track_id": ["A"]
    })

    result = add_labels(feature_df, held_out)

    assert result.loc[result["track_id"] == "B", "label"].iloc[0] == 0


# add_labels() handles duplicate held-out pairs correctly
def test_add_labels_duplicate_held_out_pairs_do_not_affect_labels():
    feature_df = pd.DataFrame({
        "user_id": [1, 1],
        "track_id": ["A", "B"]
    })

    held_out = pd.DataFrame({
        "user_id": [1, 1],
        "track_id": ["A", "A"]
    })

    result = add_labels(feature_df, held_out)

    assert result["label"].tolist() == [1, 0]


# add_labels() is not modifying the original DataFrame
def test_original_feature_dataframe_is_not_modified():
    feature_df = pd.DataFrame({
        "user_id": [1, 1],
        "track_id": ["A", "B"],
        "score": [0.5, 0.7]
    })

    original = feature_df.copy()

    held_out = pd.DataFrame({
        "user_id": [1],
        "track_id": ["A"]
    })

    add_labels(feature_df, held_out)

    pd.testing.assert_frame_equal(feature_df, original)

