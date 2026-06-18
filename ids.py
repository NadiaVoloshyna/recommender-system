from utils import make_id
import pandas as pd


def build_artists_df(recent_tracks_df, top_tracks_df, top_artists_df):
    artists_from_recent = recent_tracks_df["artist_name"]
    artists_from_tracks = top_tracks_df["artist_name"]
    artists_from_top = top_artists_df["artist_name"]

    all_artists = pd.concat([artists_from_top, artists_from_recent, artists_from_tracks ], ignore_index=True)
    unique_artists = all_artists.dropna().drop_duplicates()

    artists_df = pd.DataFrame({"artist_name": unique_artists})
    artists_df["artist_id"] = artists_df["artist_name"].apply(lambda x: make_id(x, "artist"))
    artists_df = artists_df[["artist_name", "artist_id"]]

    return artists_df


def build_tracks_df(recent_tracks_df, top_tracks_df, artists_df):
    tracks_from_recent = recent_tracks_df[["artist_name", "track_name"]]
    tracks_from_top = top_tracks_df[["artist_name", "track_name"]]

    all_tracks = pd.concat([tracks_from_recent, tracks_from_top], ignore_index=True)
    unique_tracks = all_tracks.dropna().drop_duplicates()

    # Join with artists_df to get artist_id
    tracks_df = unique_tracks.merge(artists_df, on="artist_name", how="left")
    tracks_df = tracks_df.dropna(subset=["artist_id"])  # handling for failed joins

    tracks_df["track_id"] = tracks_df["track_name"].apply(lambda x: make_id(x, "track"))
    tracks_df = tracks_df[["track_name", "artist_name", "track_id", "artist_id"]]

    return tracks_df
