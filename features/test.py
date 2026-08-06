import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from features.interaction_builder import build_interaction_dataframe
from features.embedding_builder import build_track_embeddings


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

