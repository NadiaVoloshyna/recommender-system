import pandas as pd
import pytest
from candidates.generate_candidates import \
    get_history_tracks, \
    get_similar_track_candidates, \
    get_similar_artist_candidates, \
    get_vector_candidates
import numpy as np
import faiss


@pytest.fixture
def interaction_df():
    return pd.DataFrame({
        "user_id": ["user1", "user1", "user2"],
        "track_id": ["track1", "track2", "track3"],
        "interaction_strength": [5, 3, 4]
    })


@pytest.fixture
def history_tracks():
    return pd.DataFrame({
        "track_id": ["track_1", "track_2"],
        "interaction_strength": [5.0, 3.0],
    })


@pytest.fixture
def tracks_df():
    return pd.DataFrame({
        "track_id": ["track_1", "track_2", "track_3", "track_4", "track_5", "track_6", "track_7"],
        "artist_id": ["artist_1", "artist_2", "artist_3", "artist_3", "artist_4", "artist_4", "artist_5"]})


@pytest.fixture
def artist_similarity_df():
    return pd.DataFrame({
        "artist_id": ["artist_1", "artist_1", "artist_1", "artist_1", "artist_2"],
        "similar_artist_id": ["artist_3", "artist_4", "artist_5", "artist_6", "artist_3"],
        "similarity_score": [0.95, 0.90, 0.80, 0.60, 0.85]})


# get_history_tracks() returns only the requested user's tracks
def test_get_history_tracks_returns_users_history(interaction_df):
    result = get_history_tracks("user1", interaction_df)

    expected = pd.DataFrame({
        "user_id": ["user1", "user1"],
        "track_id": ["track1", "track2"],
        "interaction_strength": [5, 3]
    })

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


# get_history_tracks() returns empty for unknown_user
def test_get_history_tracks_returns_empty_for_unknown_user(interaction_df):
    result = get_history_tracks("unknown_user", interaction_df)

    assert result.empty
    assert list(result.columns) == [
        "user_id",
        "track_id",
        "interaction_strength"
    ]


# get_similar_track_candidates() returns top k_sim candidates
def test_get_similar_track_candidates_returns_top_k():
    history_tracks = pd.DataFrame(
        {"track_id": ["track1"], "source_track_id": ["track1"], "interaction_strength": [5], })
    track_similarity_df = pd.DataFrame({
        "track_id": ["track1", "track1", "track1"],
        "similar_track_id": ["track10", "track11", "track12"],
        "similarity_score": [0.95, 0.85, 0.75]
    })

    result = get_similar_track_candidates(
        user_id="user1",
        history_tracks=history_tracks,
        track_similarity_df=track_similarity_df,
        k_sim=2)

    assert len(result) == 2
    assert result["track_id"].tolist() == ["track10", "track11"]


# get_similar_track_candidates() formats candidates correctly
def test_get_similar_track_candidates_formats_correctly():
    history_tracks = pd.DataFrame(
        {"track_id": ["track1"], "source_track_id": ["track1"], "interaction_strength": [5], })
    track_similarity_df = pd.DataFrame({
        "track_id": ["track1"],
        "similar_track_id": ["track10"],
        "similarity_score": [0.95]
    })

    result = get_similar_track_candidates(
        user_id="user1",
        history_tracks=history_tracks,
        track_similarity_df=track_similarity_df,
        k_sim=1
    )

    assert list(result.columns) == ["user_id", "track_id", "source_track_id", "interaction_strength",
                                    "track_similarity_score", "artist_similarity_score",
                                    "vector_similarity_score", "source"]
    assert result.loc[0, "user_id"] == "user1"
    assert result.loc[0, "track_id"] == "track10"
    assert result.loc[0, "source_track_id"] == "track1"
    assert result.loc[0, "interaction_strength"] == 5
    assert result.loc[0, "track_similarity_score"] == 0.95
    assert pd.isna(result.loc[0, "artist_similarity_score"])
    assert result.loc[0, "source"] == "track_similarity"


# get_similar_track_candidates() returns empty DataFrame when no candidates
def test_get_similar_track_candidates_returns_empty_when_no_tracks():
    history_tracks = pd.DataFrame(
        {"track_id": ["track1"], "source_track_id": ["track1"], "interaction_strength": [5]})
    track_similarity_df = pd.DataFrame({
        "track_id": ["track2"],
        "similar_track_id": ["track10"],
        "similarity_score": [0.50]})

    result = get_similar_track_candidates(
        user_id="user1",
        history_tracks=history_tracks,
        track_similarity_df=track_similarity_df,
        k_sim=2
    )

    assert result.empty


# get_similar_artist_candidates() returns candidates
def test_get_similar_artist_candidates_returns_candidates(
    history_tracks,
    artist_similarity_df,
    tracks_df,
):
    result = get_similar_artist_candidates(
        user_id="user_1",
        history_tracks=history_tracks,
        artist_similarity_df=artist_similarity_df,
        tracks_df=tracks_df,
        k_sim=2,
        k_artists=2
    )

    assert not result.empty

    assert list(result.columns) == [
        "user_id",
        "track_id",
        "source_track_id",
        "interaction_strength",
        "track_similarity_score",
        "artist_similarity_score",
        "vector_similarity_score",
        "source",
    ]


# get_similar_artist_candidates() raises error when required columns are missing
def test_get_similar_artist_candidates_missing_column_1(
    artist_similarity_df,
    tracks_df,
):
    history = pd.DataFrame({"track_id": ["track_1"]})

    with pytest.raises(ValueError):
        get_similar_artist_candidates(
            user_id="user_1",
            history_tracks=history,
            artist_similarity_df=artist_similarity_df,
            tracks_df=tracks_df,
            k_sim=2,
            k_artists=2
        )


def test_get_similar_artist_candidates_missing_column_2(
    history_tracks,
    tracks_df,
):
    artist_similarity_df = pd.DataFrame({
        "artist_id": ["artist_1"],
        "similar_artist_id": ["artist_3"],
        # missing similarity_score
    })

    with pytest.raises(ValueError):
        get_similar_artist_candidates(
            user_id="user_1",
            history_tracks=history_tracks,
            artist_similarity_df=artist_similarity_df,
            tracks_df=tracks_df,
            k_sim=2,
            k_artists=2
        )


def test_get_similar_artist_candidates_missing_column_3(
    history_tracks,
    artist_similarity_df,
):
    tracks_df = pd.DataFrame({
        "track_id": ["track_1"],
        # missing artist_id
    })

    with pytest.raises(ValueError):
        get_similar_artist_candidates(
            user_id="user_1",
            history_tracks=history_tracks,
            artist_similarity_df=artist_similarity_df,
            tracks_df=tracks_df,
            k_sim=2,
            k_artists=2
        )


# get_similar_artist_candidates(): k_sim validation
@pytest.mark.parametrize("bad_value", [0, -1, -10])
def test_get_similar_artist_candidates_k_sim_positive(
    history_tracks,
    artist_similarity_df,
    tracks_df,
    bad_value,
):
    with pytest.raises(ValueError, match="k_sim must be greater than 0"):
        get_similar_artist_candidates(
            user_id="user_1",
            history_tracks=history_tracks,
            artist_similarity_df=artist_similarity_df,
            tracks_df=tracks_df,
            k_sim=bad_value,
            k_artists=2
        )


@pytest.mark.parametrize("bad_value", [1.5, "10", None, True])
def test_get_similar_artist_candidates_k_sim_integer(
    history_tracks,
    artist_similarity_df,
    tracks_df,
    bad_value,
):
    with pytest.raises(TypeError, match="k_sim must be an integer"):
        get_similar_artist_candidates(
            user_id="user_1",
            history_tracks=history_tracks,
            artist_similarity_df=artist_similarity_df,
            tracks_df=tracks_df,
            k_sim=bad_value,
            k_artists=2
        )


# get_similar_artist_candidates(): k_artists validation
@pytest.mark.parametrize("bad_value", [0, -1, -10])
def test_get_similar_artist_candidates_k_artists_positive(
    history_tracks,
    artist_similarity_df,
    tracks_df,
    bad_value,
):
    with pytest.raises(ValueError, match="k_artists must be greater than 0"):
        get_similar_artist_candidates(
            user_id="user_1",
            history_tracks=history_tracks,
            artist_similarity_df=artist_similarity_df,
            tracks_df=tracks_df,
            k_sim=2,
            k_artists=bad_value
        )


@pytest.mark.parametrize("bad_value", [1.5, "10", None, True])
def test_get_similar_artist_candidates_k_artists_integer(
    history_tracks,
    artist_similarity_df,
    tracks_df,
    bad_value,
):
    with pytest.raises(TypeError, match="k_artists must be an integer"):
        get_similar_artist_candidates(
            user_id="user_1",
            history_tracks=history_tracks,
            artist_similarity_df=artist_similarity_df,
            tracks_df=tracks_df,
            k_sim=2,
            k_artists=bad_value
        )


# get_vector_candidates() converts FAISS results correctly into candidate DataFrame
def test_get_vector_candidates_returns_candidates():
    history_tracks = pd.DataFrame({"track_id": ["track_1", "track_2"], "interaction_strength": [3.0, 5.0]})

    index = faiss.IndexFlatIP(2)
    embeddings = np.array([
        [1.0, 0.0],   # track_1
        [0.9, 0.1],   # track_2
        [0.8, 0.2],   # track_3
        [0.7, 0.3],  # track_4
    ], dtype=np.float32)
    index.add(embeddings)

    track_id_mapping = ["track_1", "track_2", "track_3", "track_4"]
    track_embeddings = {
        "track_1": np.array([1.0, 0.0], dtype=np.float32),
        "track_2": np.array([0.9, 0.1], dtype=np.float32),
    }

    result = get_vector_candidates(
        user_id="user_1",
        history_tracks=history_tracks,
        faiss_index=index,
        track_id_mapping=track_id_mapping,
        track_embeddings=track_embeddings,
        k=2
    )

    assert list(result.columns) == [
        "user_id",
        "track_id",
        "source_track_id",
        "interaction_strength",
        "track_similarity_score",
        "artist_similarity_score",
        "vector_similarity_score",
        "source"
    ]
    assert result["user_id"].tolist() == ["user_1", "user_1"]
    assert result["track_id"].tolist() == ["track_3", "track_4"]
    assert result["source"].eq("vector_similarity").all()
    assert result["source_track_id"].isna().all()
    assert result["interaction_strength"].eq(4.0).all()
    assert result["track_similarity_score"].isna().all()
    assert result["artist_similarity_score"].isna().all()
    assert result["vector_similarity_score"].tolist() == pytest.approx([0.77, 0.68])


# get_vector_candidates(): k validation
@pytest.mark.parametrize("bad_k", [0, -1, -10])
def test_get_vector_candidates_rejects_non_positive_k(bad_k):
    history_tracks = pd.DataFrame({"track_id": ["track_1"]})

    with pytest.raises(ValueError, match="k must be greater than 0"):
        get_vector_candidates(
            user_id="user_1",
            history_tracks=history_tracks,
            faiss_index=None,
            track_id_mapping=[],
            track_embeddings={},
            k=bad_k
        )


@pytest.mark.parametrize("bad_k", [1.5, "10", True, False, None])
def test_get_vector_candidates_rejects_invalid_k_type(bad_k):
    history_tracks = pd.DataFrame({"track_id": ["track_1"]})

    with pytest.raises(TypeError, match="k must be an integer"):
        get_vector_candidates(
            user_id="user_1",
            history_tracks=history_tracks,
            faiss_index=None,
            track_id_mapping=[],
            track_embeddings={},
            k=bad_k
        )


# get_vector_candidates(): empty history
def test_get_vector_candidates_empty_history():
    history_tracks = pd.DataFrame(columns=["track_id"])

    result = get_vector_candidates(
        user_id="user_1",
        history_tracks=history_tracks,
        faiss_index=None,
        track_id_mapping=[],
        track_embeddings={},
        k=10
    )

    assert result.empty

    assert list(result.columns) == [
        "user_id",
        "track_id",
        "source_track_id",
        "interaction_strength",
        "track_similarity_score",
        "artist_similarity_score",
        "vector_similarity_score",
        "source"
    ]


# get_vector_candidates(): no embeddings available
def test_get_vector_candidates_no_embeddings():
    history_tracks = pd.DataFrame({"track_id": ["track_1", "track_2"]})

    result = get_vector_candidates(
        user_id="user_1",
        history_tracks=history_tracks,
        faiss_index=None,
        track_id_mapping=[],
        track_embeddings={},
        k=10
    )

    assert result.empty


# get_vector_candidates() excludes previously listened tracks
def test_get_vector_candidates_excludes_history_tracks():
    history_tracks = pd.DataFrame({"track_id": ["track_1", "track_2"], "interaction_strength": [3.0, 5.0]})

    track_embeddings = {
        "track_1": np.array([1.0, 0.0], dtype=np.float32),
        "track_2": np.array([0.9, 0.1], dtype=np.float32),
    }

    index = faiss.IndexFlatIP(2)
    embeddings = np.array([
        [1.0, 0.0],    # track_1
        [0.9, 0.1],    # track_2
        [0.8, 0.2],    # track_3
        [0.7, 0.3],    # track_4
    ], dtype=np.float32)
    index.add(embeddings)

    track_id_mapping = ["track_1", "track_2", "track_3", "track_4"]

    result = get_vector_candidates(
        user_id="user_1",
        history_tracks=history_tracks,
        faiss_index=index,
        track_id_mapping=track_id_mapping,
        track_embeddings=track_embeddings,
        k=2
    )

    assert not set(result["track_id"]).intersection({"track_1", "track_2"})

