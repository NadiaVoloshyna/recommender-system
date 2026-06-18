
def create_seeds(top_tracks_df, top_artists_df, tracks_df, artists_df):
    top_tracks_mapped = top_tracks_df.merge(tracks_df, on=["artist_name", "track_name"], how="left") # artist_name, track_name, playcount, url, user, user_id, artist_id, track_id
    top_artists_mapped = top_artists_df.merge(artists_df, on="artist_name", how="left")   # artist_name, playcount, url, user, user_id, artist_id

    seed_tracks_df = top_tracks_mapped.sort_values(["user_id", "playcount"], ascending=[True, False]).groupby("user_id").head(20)[["user_id", "track_id", "artist_id"]]
    # Handle failed joins
    seed_tracks_df = seed_tracks_df.dropna(subset=["track_id"])

    seed_artists_df = top_artists_mapped.sort_values(["user_id", "playcount"], ascending=[True, False]).groupby("user_id").head(10)[["user_id", "artist_id"]]
    seed_artists_df = seed_artists_df.dropna(subset=["artist_id"])
    # print(f"seed_tracks_df:\n{seed_tracks_df.head().to_string()}\n\n")
    # print(f"seed_artists_df:\n{seed_artists_df.head().to_string()}\n\n")

    # Get unique seeds
    unique_track_ids = seed_tracks_df[["track_id", "artist_id"]].drop_duplicates()   # track_id, artist_id
    unique_artist_ids = seed_artists_df["artist_id"].drop_duplicates().tolist()      # list of ids
    # print(f"unique_track_ids:\n{unique_track_ids[:5]}\n\n")
    # print(f"unique_artist_ids:\n{unique_artist_ids[:5]}\n\n")

    return unique_track_ids, unique_artist_ids
