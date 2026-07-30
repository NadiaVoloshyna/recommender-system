from utils import make_id
import pandas as pd
import os


def build_users_df(users_files: list) -> tuple[pd.DataFrame, dict]:
    """
    Looks at a collection of user-related file paths, finds all unique usernames from their folder names,
    assigns each user a generated ID, creates rows for the DataFrame, converts the list into a DataFrame.
    :param users_files: a list of file paths (list)
    :return: Pandas dataframe
    """
    users = sorted({
        os.path.basename(os.path.dirname(path))
        for path in users_files
    })

    rows = []
    for user in users:
        rows.append({
            "user_id": make_id(user, "user"),
            "username": user
        })

    users_df = pd.DataFrame(rows, columns=["user_id", "username"])
    user_lookup = dict(zip(
        users_df["username"],
        users_df["user_id"]
    ))

    return users_df, user_lookup


def build_artists_df(
        recent_tracks_df: pd.DataFrame,
        top_tracks_df: pd.DataFrame,
        top_artists_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """
    Extracts artist names from the artist_name columns  of the recent tracks, top tracks, and top artists dataframes.
    Combines these Series into a single Series, removes duplicates and missing values, creates a new dataframe,
    and generates a unique ID for each artist.
    :param recent_tracks_df: DataFrame containing recently played tracks.
    :param top_tracks_df: DataFrame containing users' top tracks.
    :param top_artists_df: DataFrame containing users' top artists.
    :return: DataFrame containing unique artist names and their generated IDs.
    """
    all_artists = pd.concat(
        [
            recent_tracks_df["artist_name"],
            top_tracks_df["artist_name"],
            top_artists_df["artist_name"]
        ],
        ignore_index=True,
    )

    artists_df = (
        pd.DataFrame({"artist_name": all_artists.dropna().drop_duplicates()})
        .reset_index(drop=True)
    )

    artists_df["artist_id"] = artists_df["artist_name"].map(
        lambda artist: make_id(artist, "artist")
    )

    artists_df = artists_df[["artist_id", "artist_name"]]
    artist_lookup = dict(zip(
        artists_df["artist_name"],
        artists_df["artist_id"]
    ))

    return artists_df, artist_lookup


def build_tracks_df(
        recent_tracks_df: pd.DataFrame,
        top_tracks_df: pd.DataFrame,
        artists_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """
    Build a unique track lookup table.
    Combine recent and top tracks, remove missing values and duplicates, attach artist IDs, remove tracks
    without known artists, generate unique track IDs, and select the final columns.
    :param recent_tracks_df: DataFrame containing recently played tracks.
    :param top_tracks_df: DataFrame containing users' top tracks.
    :param artists_df: DataFrame containing unique artist names and their generated IDs.
    :return: track lookup table containing: track_id, track_name, artist_id, artist_name
    """
    all_tracks = pd.concat(
        [
            recent_tracks_df[["track_name", "artist_name"]],
            top_tracks_df[["track_name", "artist_name"]]
        ],
        ignore_index=True,
    )

    tracks_df = (
        all_tracks
        .dropna(subset=["track_name", "artist_name"])
        .drop_duplicates()
        .merge(artists_df, on="artist_name", how="left", validate="many_to_one")
        .dropna(subset=["artist_id"])
        .reset_index(drop=True)
    )

    # Generate unique track IDs
    tracks_df["track_id"] = [
        make_id(f"{artist}_{track}", "track")
        for artist, track in zip(tracks_df["artist_name"], tracks_df["track_name"])
    ]
    tracks_df = tracks_df[["track_id", "track_name", "artist_id", "artist_name"]]
    track_lookup = dict(zip(
        zip(tracks_df["track_name"], tracks_df["artist_name"]),
        tracks_df["track_id"]
    ))

    return tracks_df, track_lookup


