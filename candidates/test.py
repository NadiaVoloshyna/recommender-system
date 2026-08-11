import pandas as pd
import pytest
from candidates.generate_candidates import get_history_tracks, get_similar_track_candidates


@pytest.fixture
def interaction_df():
    return pd.DataFrame({
        "user_id": ["user1", "user1", "user2"],
        "track_id": ["track1", "track2", "track3"],
        "interaction_strength": [5, 3, 4]
    })


# get_history_tracks() returns only the requested user's tracks
def test_get_history_tracks_returns_users_history(interaction_df):
    result = get_history_tracks("user1", interaction_df)

    expected = pd.DataFrame({
        "user_id": ["user1", "user1"],
        "track_id": ["track1", "track2"],
        "interaction_strength": [5, 3],
        "similarity_score": [None, None],
        "source": ["history", "history"], })

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


# get_history_tracks() returns empty for unknown_user
def test_get_history_tracks_returns_empty_for_unknown_user(interaction_df):
    result = get_history_tracks("unknown_user", interaction_df)

    assert result.empty
    assert list(result.columns) == [
        "user_id",
        "track_id",
        "interaction_strength",
        "similarity_score",
        "source",
    ]


# get_history_tracks() adds similarity_score with None
def test_get_history_tracks_similarity_score_is_none(interaction_df):
    result = get_history_tracks("user1", interaction_df)

    assert result["similarity_score"].isna().all()


# get_history_tracks() adds source with "history"
def test_get_history_tracks_source_is_history(interaction_df):
    result = get_history_tracks("user1", interaction_df)

    assert (result["source"] == "history").all()


# get_similar_track_candidates() applies similarity threshold
def test_get_similar_track_candidates_applies_threshold():
    history_candidates = pd.DataFrame({
        "user_id": ["user1", "user1"],
        "track_id": ["track1", "track2"],
        "interaction_strength": [5, 3],
        "similarity_score": [None, None],
        "source": ["history", "history"],
    })

    track_similarity_df = pd.DataFrame({
        "track_id": ["track1", "track1", "track1", "track2", "track2"],
        "similar_track_id": ["track10", "track11", "track12", "track20", "track21"],
        "similarity_score": [0.95, 0.85, 0.60, 0.90, 0.70]
    })

    result = get_similar_track_candidates(
        user_id="user1",
        history_candidates=history_candidates,
        track_similarity_df=track_similarity_df,
        k_sim=2,
        similarity_threshold=0.7
    )

    assert result["track_id"].tolist() == ["track10", "track11", "track20", "track21"]
    assert result["similarity_score"].tolist() == [0.95, 0.85, 0.90, 0.70]


# get_similar_track_candidates() returns top k_sim candidates
def test_get_similar_track_candidates_returns_top_k():
    history_candidates = pd.DataFrame({"track_id": ["track1"]})
    track_similarity_df = pd.DataFrame({
        "track_id": ["track1", "track1", "track1"],
        "similar_track_id": ["track10", "track11", "track12"],
        "similarity_score": [0.95, 0.85, 0.75]
    })

    result = get_similar_track_candidates(
        user_id="user1",
        history_candidates=history_candidates,
        track_similarity_df=track_similarity_df,
        k_sim=2,
        similarity_threshold=0.0)

    assert len(result) == 2
    assert result["track_id"].tolist() == ["track10", "track11"]


# get_similar_track_candidates() formats candidates correctly
def test_get_similar_track_candidates_formats_correctly():
    history_candidates = pd.DataFrame({"track_id": ["track1"]})
    track_similarity_df = pd.DataFrame({
        "track_id": ["track1"],
        "similar_track_id": ["track10"],
        "similarity_score": [0.95]
    })

    result = get_similar_track_candidates(
        user_id="user1",
        history_candidates=history_candidates,
        track_similarity_df=track_similarity_df,
        k_sim=1,
        similarity_threshold=0.7
    )

    assert list(result.columns) == ["user_id", "track_id", "interaction_strength", "similarity_score", "source"]
    assert result.loc[0, "user_id"] == "user1"
    assert result.loc[0, "track_id"] == "track10"
    assert pd.isna(result.loc[0, "interaction_strength"])
    assert result.loc[0, "similarity_score"] == 0.95
    assert result.loc[0, "source"] == "track_similarity"


# get_similar_track_candidates() returns empty DataFrame when no candidates
def test_returns_empty_when_no_similar_tracks():
    history_candidates = pd.DataFrame({"track_id": ["track1"]})
    track_similarity_df = pd.DataFrame({
        "track_id": ["track1"],
        "similar_track_id": ["track10"],
        "similarity_score": [0.50]})

    result = get_similar_track_candidates(
        user_id="user1",
        history_candidates=history_candidates,
        track_similarity_df=track_similarity_df,
        k_sim=2,
        similarity_threshold=0.70
    )

    assert result.empty
    assert list(result.columns) == ["user_id", "track_id", "interaction_strength", "similarity_score", "source"]






