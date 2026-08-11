import pandas as pd
import pytest
from candidates.generate_candidates import get_history_tracks


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