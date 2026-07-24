import os
import json
import pandas as pd
from utils import make_id


def get_users_data_paths(base_path: str = "raw_data") -> list:
    """
    Creates an empty list, loops through folders inside raw_data, builds the full user folder path,
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


def build_user_dataframes(saved_files: list, user_lookup: dict) -> dict:
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


def load_similarity_data(base_path):
    all_files = []

    for file in os.listdir(base_path):
        file_path = os.path.join(base_path, file)
        all_files.append(file_path)

    return all_files


def build_similarity_dataframes(group, saved_files):
    dfs = []

    for file_path in saved_files:
        item_id = os.path.splitext(os.path.basename(file_path))[0]

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        similar_items = data.get(f"similar{group}s", {}).get(f"{group}", [])
        if not similar_items:
            continue

        df = pd.json_normalize(similar_items)

        # Rename raw API fields
        df = df.rename(columns={"name": f"similar_{group}_name", "match": "similarity_score"})

        # Filter invalid rows
        df = df[df[f"similar_{group}_name"].notna() & df["similarity_score"].notna()]

        # Add metadata
        df[f"{group}_id"] = item_id

        # Enforce schema
        df = df[[f"{group}_id", f"similar_{group}_name", "similarity_score"]]

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

