from utils import make_id


def map_similarity_ids(tracks_similarity_df, artists_similarity_df, tracks_df, artists_df):
    tracks_similarity_df = tracks_similarity_df.merge(                                        # track_id, similarity_score, similar_track_id, similar_track_name, track_name
        tracks_df[["track_id", "track_name"]],
        left_on="similar_track_name",
        right_on="track_name",
        how="left"
    ).rename(columns={"track_id_x": "track_id", "track_id_y": "similar_track_id"})

    missing = tracks_similarity_df["similar_track_id"].isna()
    tracks_similarity_df.loc[missing, "similar_track_id"] = (
        tracks_similarity_df.loc[missing, "similar_track_name"]
        .apply(lambda x: make_id(x, "track"))
    )
    tracks_similarity_df = tracks_similarity_df.drop(columns=["similar_track_name", "track_name"])

    artists_similarity_df = artists_similarity_df.merge(                                      # artist_id, similarity_score, similar_artist_id, similar_artist_name, artist_name
        artists_df[["artist_id", "artist_name"]],
        left_on="similar_artist_name",
        right_on="artist_name",
        how="left"
    ).rename(columns={"artist_id_x": "artist_id", "artist_id_y": "similar_artist_id"})

    missing = artists_similarity_df["similar_artist_id"].isna()
    artists_similarity_df.loc[missing, "similar_artist_id"] = (
        artists_similarity_df.loc[missing, "similar_artist_name"]
        .apply(lambda x: make_id(x, "artist"))
    )
    artists_similarity_df = artists_similarity_df.drop(columns=["similar_artist_name", "artist_name"])

    return tracks_similarity_df, artists_similarity_df
