import pandas as pd
import numpy as np
from features.utils import validate_columns


def build_interaction_dataframe(
        recent_tracks_df: pd.DataFrame,
        top_tracks_df: pd.DataFrame,
        tracks_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Creates a user–track interaction DataFrame by combining users' recently played tracks and top tracks into
    a single implicit feedback dataset. Each interaction is assigned a weight reflecting the strength of the user's
    preference. Multiple interactions for the same user–track pair are aggregated into a single interaction score.
    :param recent_tracks_df: recently played tracks for each user (pd.DataFrame)
    :param top_tracks_df: users' top tracks together with play counts (pd.DataFrame)
    :param tracks_df: track lookup table containing global track IDs and track metadata (pd.DataFrame)
    :return: pd.DataFrame containing: user_id, track_id, interaction_strength
    """
    validate_columns(
        recent_tracks_df,
        ["user_id", "track_name", "artist_name"],
        "recent_tracks_df"
    )

    validate_columns(
        top_tracks_df,
        ["user_id", "track_name", "artist_name", "playcount"],
        "top_tracks_df"
    )

    validate_columns(
        tracks_df,
        ["track_id", "track_name", "artist_name"],
        "tracks_df"
    )

    if recent_tracks_df.empty and top_tracks_df.empty:
        raise ValueError("No interaction data available.")

    # Match recent tracks with the track lookup table to obtain internal IDs
    recent_tracks_mapped = recent_tracks_df.merge(
        tracks_df,
        on=["track_name", "artist_name"],
        how="left",
        validate="many_to_one")

    # Discard interactions that could not be mapped to a track
    recent_tracks_mapped = recent_tracks_mapped.dropna(subset=["track_id"]).copy()

    if not recent_tracks_df.empty:
        recent_match_rate = len(recent_tracks_mapped) / len(recent_tracks_df)
        if recent_match_rate < 0.8:
            print(
                f"Warning: only {recent_match_rate:.1%} of recent tracks matched."
            )

    # Assign implicit-feedback weight
    recent_tracks_mapped["weight"] = 1

    # Match top tracks with the track lookup table
    top_tracks_mapped = top_tracks_df.merge(
        tracks_df,
        on=["track_name", "artist_name"],
        how="left",
        validate="many_to_one")

    # Discard unmatched tracks
    top_tracks_mapped = top_tracks_mapped.dropna(subset=["track_id"]).copy()

    if not top_tracks_df.empty:
        top_match_rate = len(top_tracks_mapped) / len(top_tracks_df)
        if top_match_rate < 0.8:
            print(
                f"Warning: only {top_match_rate:.1%} of top tracks matched."
            )

    # Assign implicit-feedback weight
    top_tracks_mapped["weight"] = np.log1p(top_tracks_mapped["playcount"])

    recent_tracks_mapped = recent_tracks_mapped[["user_id", "track_id", "weight"]]
    top_tracks_mapped = top_tracks_mapped[["user_id", "track_id", "weight"]]

    # Combine interaction events
    interaction_events = pd.concat(
        [recent_tracks_mapped, top_tracks_mapped],
        ignore_index=True
    )

    # Aggregate interaction events for each user–track pair by summing the implicit-feedback weights
    interaction_df = (
        interaction_events
        .groupby(["user_id", "track_id"], as_index=False)
        .agg(
            interaction_strength=("weight", "sum")
             )
    )

    if interaction_df.empty:
        raise ValueError("Interaction dataframe is empty.")

    return interaction_df


