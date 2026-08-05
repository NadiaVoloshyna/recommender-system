import os
import json
import pandas as pd
from etl.utils import make_id
from pathlib import Path
from config.paths import LISTENING_HISTORY_DIR


def get_users_data_paths(base_path: Path = LISTENING_HISTORY_DIR) -> list:
    """
    Creates an empty list, loops through folders inside listening_history, builds the full user folder path,
    checks if it is actually a folder, loops through files inside the user folder, create full file path,
    adds the path to the list.
    :param base_path: folder containing stored user data (str, optional)
    :return: a list of all files stored inside user folders (list)
    """
    all_files = []

    for user_folder in os.listdir(base_path):
        user_path = os.path.join(base_path, user_folder)
        if not os.path.isdir(user_path):
            continue

        for file in os.listdir(user_path):
            file_path = os.path.join(user_path, file)
            all_files.append(file_path)

    return all_files


def build_user_dataframes(saved_files: list, user_lookup: dict) -> dict[str, pd.DataFrame]:
    """
    Builds cleaned pandas DataFrames from Last.fm JSON files.
    Creates storage for the three datasets, loops through every JSON file, gets the API name and user,
    loads and processes the JSON, combines all users.
    :param saved_files: a list of saved JSON files from different users (list)
    :param user_lookup: a dictionary containing unique users and their generated ids (dict)
    :return: a dictionary of combined dataframes: recent_tracks, top_tracks, and top_artists
    """
    # Store DataFrames separately by Last.fm endpoint before combining all users
    df_groups = {
        "recent_tracks": [],
        "top_tracks": [],
        "top_artists": []
    }

    for file_path in saved_files:
        # Extract API endpoint name and user identity from file path
        name = os.path.splitext(os.path.basename(file_path))[0]
        user = os.path.basename(os.path.dirname(file_path))
        user_id = user_lookup[user]

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if name == "user.getRecentTracks":
            recent_tracks = data.get("recenttracks", {}).get("track", [])
            if not recent_tracks:
                continue

            rows = []
            for track in recent_tracks:
                rows.append({
                    "track_name": track.get("name"),
                    "artist_name": track.get("artist", {}).get("#text"),
                    "timestamp": track.get("date", {}).get("uts")
                })

            df = pd.DataFrame(rows)

            df = df[
                df["track_name"].notna() &
                df["artist_name"].notna() &
                df["timestamp"].notna() &
                (df["track_name"].str.strip().str.len() > 0) &
                (df["artist_name"].str.strip().str.len() > 0)
                ]
            df = df.assign(user_id=user_id, user=user)
            df = df.reindex(columns=["user_id", "user", "track_name", "artist_name", "timestamp"])

            df_groups["recent_tracks"].append(df)

        elif name == "user.getTopTracks":
            top_tracks = data.get("toptracks", {}).get("track", [])
            if not top_tracks:
                continue

            rows = []
            for track in top_tracks:
                rows.append({
                    "track_name": track.get("name"),
                    "artist_name": track.get("artist", {}).get("name"),
                    "playcount": int(track.get("playcount") or 0)
                })

            df = pd.DataFrame(rows)

            df = df[
                df["track_name"].notna() &
                df["artist_name"].notna() &
                (df["track_name"].str.strip().str.len() > 0) &
                (df["artist_name"].str.strip().str.len() > 0)
                ]
            df = df.assign(user_id=user_id, user=user)
            df = df.reindex(columns=["user_id", "user", "track_name", "artist_name", "playcount"])

            df_groups["top_tracks"].append(df)

        elif name == "user.getTopArtists":
            top_artists = data.get("topartists", {}).get("artist", [])
            if not top_artists:
                continue

            rows = []
            for artist in top_artists:
                rows.append({
                    "artist_name": artist.get("name"),
                    "playcount": int(artist.get("playcount") or 0),
                    "rank": int(artist.get("@attr", {}).get("rank") or 0)
                })

            df = pd.DataFrame(rows)

            df = df[
                df["artist_name"].notna() &
                (df["artist_name"].str.strip().str.len() > 0) &
                (df["playcount"] > 0)
                ]
            df = df.assign(user_id=user_id, user=user)
            df = df.reindex(columns=["user_id", "user", "artist_name", "playcount", "rank"])

            df_groups["top_artists"].append(df)

    # Combine DataFrames from all users into one DataFrame per dataset type
    result = {
        key: pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
        for key, df_list in df_groups.items()
    }

    return result


def load_similarity_data(base_path: str) -> list:
    """
    Scans a directory and returns the paths of all stored similarity data files for later loading and processing.
    :param base_path: directory containing stored similarity files (str)
    :return: a list of file paths contained in the directory
    Returns an empty list if the directory does not exist or contains no JSON files.
    """
    if not os.path.exists(base_path):
        return []

    return [
        os.path.join(base_path, file)
        for file in os.listdir(base_path)
    ]


def build_track_similarity_dataframe(saved_files: list[str], track_lookup: dict) -> pd.DataFrame:
    """
    Builds a DataFrame containing similarity relationships between tracks from a collection
    of JSON files.
    The function reads each JSON file, extracts the original track information and its
    similar tracks, validates the extracted data, and combines the results into a single
    DataFrame.
    :param saved_files: a list of file paths to JSON files containing similar track data (list[str])
    :param track_lookup: a dictionary mapping (track_name, artist_name) tuples to track IDs (dict)
    :return: a dataframe containing the original track ID and name, the similar track ID and name, and the similarity
    score (pd.DataFrame). If no valid data is found, an empty DataFrame with the expected columns is returned.
    """
    dfs = []

    for file_path in saved_files:
        # The filename contains the ID of the original track
        track_id = os.path.splitext(os.path.basename(file_path))[0]

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        track_name = data.get("similartracks", {}).get("@attr", {}).get("track")
        similar_tracks = data.get("similartracks", {}).get("track", [])
        if not similar_tracks:
            continue

        rows = []
        for track in similar_tracks:
            try:
                score = float(track.get("match"))
            except (TypeError, ValueError):
                score = 0.0

            similar_track_name = track.get("name")
            if not similar_track_name:
                continue

            similar_artist_name = track.get("artist", {}).get("name")
            # Convert track name and artist name into an existing track ID
            similar_track_id = track_lookup.get((similar_track_name, similar_artist_name))

            if similar_track_id is None:
                similar_track_id = make_id(
                    f"{similar_track_name}_{similar_artist_name}",
                    "track"
                )
                track_lookup[(similar_track_name, similar_artist_name)] = similar_track_id

            rows.append({
                "similar_track_id": similar_track_id,
                "similar_track_name": similar_track_name,
                "similarity_score": score
            })

        df = pd.DataFrame(rows)

        df = df[
            df["similar_track_id"].notna()
            & df["similar_track_name"].notna()
            & (df["similar_track_name"].str.strip().str.len() > 0)
            & (df["similarity_score"] > 0)
            ]
        df = df.assign(track_id=track_id, track_name=track_name)
        df = df[["track_id", "track_name", "similar_track_id", "similar_track_name", "similarity_score"]]

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(
        columns=[
            "track_id",
            "track_name",
            "similar_track_id",
            "similar_track_name",
            "similarity_score",
        ]
    )


def build_artist_similarity_dataframe(saved_files: list[str], artist_lookup: dict) -> pd.DataFrame:
    """
    Processes each saved JSON file by extracting the original artist and their similar artists,
    creating rows containing the similar artist IDs, names, and similarity scores,
    removing invalid records, adding the original artist information,
    and combining the results from all files into a single DataFrame.
    :param saved_files: a list of file paths to JSON files containing similar artist data (list[str])
    :param artist_lookup: a dictionary mapping artist names to artist IDs (dict)
    :return: a dataframe containing the original artist ID and name, the similar artist ID and name,
    and the similarity score (pd.DataFrame).
    If no valid data is found, an empty DataFrame with the expected columns is returned.
    """
    dfs = []

    for file_path in saved_files:
        # The filename contains the ID of the original artist
        artist_id = os.path.splitext(os.path.basename(file_path))[0]

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        artist_name = data.get("similarartists", {}).get("@attr", {}).get("artist")
        similar_artists = data.get("similarartists", {}).get("artist", [])
        if not similar_artists:
            continue

        rows = []
        for artist in similar_artists:
            try:
                score = float(artist.get("match"))
            except (TypeError, ValueError):
                score = 0.0

            similar_artist_name = artist.get("name")
            if not similar_artist_name:
                continue

            # Convert artist name into an existing artist ID
            similar_artist_id = artist_lookup.get(similar_artist_name)
            if similar_artist_id is None:
                similar_artist_id = make_id(similar_artist_name, "artist")
                artist_lookup[similar_artist_name] = similar_artist_id

            rows.append({
                "similar_artist_id": similar_artist_id,
                "similar_artist_name": similar_artist_name,
                "similarity_score": score
            })

        df = pd.DataFrame(rows)

        df = df[
            df["similar_artist_id"].notna()
            & df["similar_artist_name"].notna()
            & (df["similar_artist_name"].str.strip().str.len() > 0)
            & (df["similarity_score"] > 0)
            ]
        df = df.assign(artist_id=artist_id, artist_name=artist_name)
        df = df[["artist_id", "artist_name", "similar_artist_id", "similar_artist_name", "similarity_score"]]

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(
        columns=[
            "artist_id",
            "artist_name",
            "similar_artist_id",
            "similar_artist_name",
            "similarity_score",
        ]
    )


