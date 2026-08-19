import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from features.interaction_builder import build_interaction_dataframe
from features.embedding_builder import build_track_embeddings
from features.candidate_features import build_candidate_features


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
