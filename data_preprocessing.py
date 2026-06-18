import os
import json
import pandas as pd
from utils import make_id


def load_user_data(base_path="raw_data"):
    all_files = []

    for user_folder in os.listdir(base_path):
        user_path = os.path.join(base_path, user_folder)
        if not os.path.isdir(user_path):
            continue

        for file in os.listdir(user_path):
            file_path = os.path.join(user_path, file)
            all_files.append(file_path)

    return all_files


def load_similarity_data(base_path):
    all_files = []

    for file in os.listdir(base_path):
        file_path = os.path.join(base_path, file)
        all_files.append(file_path)

    return all_files


def build_user_dataframes(saved_files):
    dfs = {}
    user_ids = {}

    for file_path in saved_files:
        name = os.path.splitext(os.path.basename(file_path))[0]
        user = file_path.split(os.sep)[-2]
        if user in user_ids:
            user_id = user_ids[user]
        else:
            user_id = make_id(user, "user")
            user_ids[user] = user_id

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if name == "user.getRecentTracks":
            recent_tracks = data.get("recenttracks", {}).get("track", [])
            if not recent_tracks:
                continue

            rows = []
            for t in recent_tracks:
                artist = t.get("artist")

                if isinstance(artist, dict):
                    artist_name = (artist.get("name") or artist.get("#text"))
                elif isinstance(artist, str):
                    artist_name = artist
                else:
                    artist_name = None

                rows.append({
                    "artist_name": artist_name,
                    "track_name": t.get("name"),
                    "timestamp": t.get("date", {}).get("uts"),
                    "url": t.get("url")
                })
            df = pd.DataFrame(rows)

            # Filter invalid rows
            df = df[
                df["artist_name"].notna() &
                df["track_name"].notna() &
                (df["artist_name"].str.len() > 0) &
                (df["track_name"].str.len() > 0)
                ]

            # Enforce schema
            df = df.reindex(columns=["track_name", "artist_name", "timestamp", "url"])

            # Add metadata
            df = df.assign(user=user, user_id=user_id)

            dfs.setdefault("recent_tracks", []).append(df)

        elif name == "user.getTopTracks":
            top_tracks = data.get("toptracks", {}).get("track", [])
            if not top_tracks:
                continue

            df = pd.json_normalize(top_tracks)

            # Rename raw API fields
            df = df.rename(columns={"artist.name": "artist_name", "name": "track_name"})

            # Filter invalid rows
            df = df[
                df["artist_name"].notna() &
                df["track_name"].notna() &
                (df["artist_name"].str.len() > 0) &
                (df["track_name"].str.len() > 0)
            ]

            # Enforce schema
            df = df.reindex(columns=["track_name", "artist_name", "playcount", "url"])

            # Add metadata
            df = df.assign(user=user, user_id=user_id)

            dfs.setdefault("top_tracks", []).append(df)

        elif name == "user.getTopArtists":
            top_artists = data.get("topartists", {}).get("artist", [])
            if not top_artists:
                continue

            df = pd.json_normalize(top_artists)

            # Rename raw API fields
            df = df.rename(columns={"name": "artist_name"})

            # Filter invalid rows
            df = df[df["artist_name"].notna() & (df["artist_name"].str.len() > 0)]

            # Enforce schema
            df = df.reindex(columns=["artist_name", "playcount", "url"])

            # Add metadata
            df = df.assign(user=user, user_id=user_id)

            dfs.setdefault("top_artists", []).append(df)

    # Combine all users
    result = {}
    for key, df_list in dfs.items():
        result[key] = pd.concat(df_list, ignore_index=True)

    return result


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

