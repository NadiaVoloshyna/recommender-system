import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from features.interaction_builder import build_interaction_dataframe


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

