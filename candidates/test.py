import pandas as pd
import pytest
from candidates.generate_candidates import \
    get_history_tracks, \
    get_similar_track_candidates, \
    get_similar_artist_candidates


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


# get_similar_track_candidates() applies similarity threshold
def test_get_similar_track_candidates_applies_threshold():
    history_tracks = pd.DataFrame({
        "user_id": ["user1", "user1"],
        "track_id": ["track1", "track2"],
        "source_track_id": ["track1", "track2"],
        "interaction_strength": [5, 3],
        "track_similarity_score": [None, None],
        "artist_similarity_score": [None, None],
        "source": ["history", "history"],
    })

    track_similarity_df = pd.DataFrame({
        "track_id": ["track1", "track1", "track1", "track2", "track2"],
        "similar_track_id": ["track10", "track11", "track12", "track20", "track21"],
        "similarity_score": [0.95, 0.85, 0.60, 0.90, 0.70]
    })

    result = get_similar_track_candidates(
        user_id="user1",
        history_tracks=history_tracks,
        track_similarity_df=track_similarity_df,
        k_sim=2,
        similarity_threshold=0.7
    )

    assert result["track_id"].tolist() == ["track10", "track11", "track20", "track21"]
    assert result["track_similarity_score"].tolist() == [0.95, 0.85, 0.90, 0.70]


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
        k_sim=2,
        similarity_threshold=0.0)

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
        k_sim=1,
        similarity_threshold=0.7
    )

    assert list(result.columns) == ["user_id", "track_id", "source_track_id", "interaction_strength",
                                    "track_similarity_score", "artist_similarity_score", "source"]
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
        {"track_id": ["track1"], "source_track_id": ["track1"], "interaction_strength": [5], })
    track_similarity_df = pd.DataFrame({
        "track_id": ["track1"],
        "similar_track_id": ["track10"],
        "similarity_score": [0.50]})

    result = get_similar_track_candidates(
        user_id="user1",
        history_tracks=history_tracks,
        track_similarity_df=track_similarity_df,
        k_sim=2,
        similarity_threshold=0.70
    )

    assert result.empty
    assert list(result.columns) == ["user_id", "track_id", "source_track_id", "interaction_strength",
                                    "track_similarity_score", "artist_similarity_score", "source"]


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
        k_artists=2,
        similarity_threshold=0.70,
    )

    assert not result.empty

    assert list(result.columns) == [
        "user_id",
        "track_id",
        "source_track_id",
        "interaction_strength",
        "track_similarity_score",
        "artist_similarity_score",
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
            k_artists=2,
            similarity_threshold=0.70,
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
            k_artists=2,
            similarity_threshold=0.70,
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
            k_artists=2,
            similarity_threshold=0.70,
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
            k_artists=2,
            similarity_threshold=0.70,
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
            k_artists=2,
            similarity_threshold=0.70,
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
            k_artists=bad_value,
            similarity_threshold=0.70,
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
            k_artists=bad_value,
            similarity_threshold=0.70,
        )


# get_similar_artist_candidates(): similarity threshold validation
@pytest.mark.parametrize("bad_value", [-0.1, 1.1, 2, -1])
def test_get_similar_artist_candidates_similarity_threshold__between_zero_and_one(
    history_tracks,
    artist_similarity_df,
    tracks_df,
    bad_value,
):
    with pytest.raises(
        ValueError,
        match="similarity_threshold must be between 0 and 1",
    ):
        get_similar_artist_candidates(
            user_id="user_1",
            history_tracks=history_tracks,
            artist_similarity_df=artist_similarity_df,
            tracks_df=tracks_df,
            k_sim=2,
            k_artists=2,
            similarity_threshold=bad_value,
        )


@pytest.mark.parametrize("bad_value", ["0.70", None, True])
def test_get_similar_artist_candidates_similarity_threshold_numeric(
    history_tracks,
    artist_similarity_df,
    tracks_df,
    bad_value,
):
    with pytest.raises(
        TypeError,
        match="similarity_threshold must be numeric",
    ):
        get_similar_artist_candidates(
            user_id="user_1",
            history_tracks=history_tracks,
            artist_similarity_df=artist_similarity_df,
            tracks_df=tracks_df,
            k_sim=2,
            k_artists=2,
            similarity_threshold=bad_value,
        )


# get_similar_artist_candidates() threshold filters artists
def test_get_similar_artist_candidates_similarity_threshold_filters_artists(
    history_tracks,
    artist_similarity_df,
    tracks_df,
):
    result = get_similar_artist_candidates(
        user_id="user_1",
        history_tracks=history_tracks,
        artist_similarity_df=artist_similarity_df,
        tracks_df=tracks_df,
        k_sim=10,
        k_artists=10,
        similarity_threshold=0.70,
    )

    # artist_6 has similarity 0.60 and should be excluded
    assert (result["artist_similarity_score"] >= 0.70).all()


def test_get_similar_artist_candidates_below_threshold_artist_track_is_excluded(
    history_tracks,
    artist_similarity_df,
    tracks_df,
):
    tracks_df = pd.concat(
        [
            tracks_df,
            pd.DataFrame({
                "track_id": ["track_8"],
                "artist_id": ["artist_6"],
            }),
        ],
        ignore_index=True,
    )

    result = get_similar_artist_candidates(
        user_id="user_1",
        history_tracks=history_tracks,
        artist_similarity_df=artist_similarity_df,
        tracks_df=tracks_df,
        k_sim=10,
        k_artists=10,
        similarity_threshold=0.70,
    )

    assert "track_8" not in result["track_id"].values


