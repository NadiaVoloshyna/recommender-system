import pandas as pd


def create_seeds(
        top_tracks_df: pd.DataFrame,
        top_artists_df: pd.DataFrame,
        tracks_df: pd.DataFrame,
        artists_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Creates unique seed tables for similarity API requests.
    Maps top tracks and top artists to internal IDs, selects each user's top 20 tracks and top 10 artists
    based on play count, removes duplicate seeds across users, and returns the unique seed tables.
    :param top_tracks_df: dataframe containing users' top tracks.
    :param top_artists_df: dataframe containing users' top artists.
    :param tracks_df: dataframe containing the track lookup table with generated track IDs and artist IDs.
    :param artists_df: dataframe containing the artist lookup table with generated artist IDs.
    :return:
        seed_tracks: DataFrame containing unique seed tracks with track_id, track_name, artist_id, and artist_name
        seed_artists: DataFrame containing unique seed artists with artist_id and artist_name
    """
    top_tracks_mapped = top_tracks_df.merge(
        tracks_df,
        on=["artist_name", "track_name"],
        how="left",
        validate="many_to_one",
    )

    top_artists_mapped = top_artists_df.merge(
        artists_df,
        on="artist_name",
        how="left",
        validate="many_to_one",
    )

    seed_tracks_df = (
        top_tracks_mapped
        .sort_values(["user_id", "playcount"], ascending=[True, False])
        .groupby("user_id")
        .head(20)[["track_id", "track_name", "artist_id", "artist_name"]]
        .dropna(subset=["track_id"])
    )

    seed_artists_df = (
        top_artists_mapped
        .sort_values(["user_id", "playcount"], ascending=[True, False])
        .groupby("user_id")
        .head(10)[["artist_id", "artist_name"]]
        .dropna(subset=["artist_id"])
    )

    seed_tracks = seed_tracks_df.drop_duplicates(subset=["track_id"])
    seed_artists = seed_artists_df.drop_duplicates(subset=["artist_id"])

    return seed_tracks, seed_artists

