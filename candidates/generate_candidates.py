import pandas as pd


def get_history_tracks(user_id, interaction_df):
    return (
        interaction_df[
            interaction_df["user_id"] == user_id
        ][["user_id", "track_id", "interaction_strength"]]
        .drop_duplicates()
        .assign(
            similarity_score=None,
            source="history"
        ))


def get_similar_track_candidates(user_id, history_candidates, track_similarity_df, k_sim, similarity_threshold):
    candidates = []

    for track_id in history_candidates["track_id"]:
        sims = (
            track_similarity_df[
                (track_similarity_df["track_id"] == track_id)
                & (track_similarity_df["similarity_score"] >= similarity_threshold)
                ]
            .sort_values("similarity_score", ascending=False)
            .head(k_sim)
            .copy()
        )

        sims = sims[["similar_track_id", "similarity_score"]].rename(
            columns={"similar_track_id": "track_id"})
        sims["user_id"] = user_id
        sims["interaction_strength"] = None
        sims["source"] = "track_similarity"

        candidates.append(sims[["user_id", "track_id", "interaction_strength", "similarity_score", "source"]])

    if not candidates:
        return pd.DataFrame(columns=["user_id", "track_id", "interaction_strength", "similarity_score", "source"])

    return pd.concat(candidates, ignore_index=True)


def get_similar_artist_candidates(
        user_id,
        interaction_df,
        artist_similarity_df,
        tracks_df,
        k_sim,
        similarity_threshold
):
    user_history = interaction_df[interaction_df["user_id"] == user_id]

    history_artists = (
        user_history
        .merge(tracks_df, on="track_id", how="left")
        [["artist_id"]]
        .drop_duplicates()
    )

    # Find similar artists for user's history artists
    similar_artists = artist_similarity_df[
        artist_similarity_df["artist_id"].isin(
            history_artists["artist_id"]
        )
    ].copy()

    # Apply similarity threshold
    similar_artists = similar_artists[similar_artists["similarity_score"] >= similarity_threshold]

    # Keep top k_sim similar artists for each source artist
    similar_artists = (
        similar_artists
        .sort_values(
            ["artist_id", "similarity_score"],
            ascending=[True, False]
        )
        .groupby("artist_id")
        .head(k_sim)
    )

    similar_artists = similar_artists[
        ["similar_artist_id", "similarity_score"]
    ].rename(
        columns={"similar_artist_id": "artist_id"}
    )

    # Connect similar artists to their tracks
    artist_tracks = tracks_df[["artist_id", "track_id"]]\
        .merge(similar_artists, on="artist_id", how="inner")

    # Remove already-listened tracks
    history_track_ids = set(user_history["track_id"])

    artist_tracks = artist_tracks[
        ~artist_tracks["track_id"].isin(history_track_ids)
    ].copy()

    # Common candidate schema
    artist_tracks["user_id"] = user_id
    artist_tracks["interaction_strength"] = None
    artist_tracks["source"] = "artist_similarity"

    return artist_tracks[["user_id", "track_id", "interaction_strength", "similarity_score", "source"]].drop_duplicates()


def generate_candidates(
        tracks_df,
        track_similarity_df,
        artist_similarity_df,
        interaction_df,
        faiss_index,
        track_id_mapping,
        k_track_sim,
        k_artist_sim,
        similarity_threshold
):
    all_candidates = []

    for user_id in interaction_df["user_id"].unique():
        history_candidates = get_history_tracks(
            user_id,
            interaction_df
        )

        track_candidates = get_similar_track_candidates(
            user_id,
            history_candidates,
            track_similarity_df,
            k_sim=k_track_sim,
            similarity_threshold=similarity_threshold
        )

        artist_tracks = get_similar_artist_candidates(
            user_id,
            interaction_df,
            artist_similarity_df,
            tracks_df,
            k_sim=k_artist_sim,
            similarity_threshold=similarity_threshold
        )

        # vector_candidates = get_vector_candidates(history_candidates, faiss_index, track_id_mapping, track_embeddings)

        candidates = pd.concat([history_candidates, track_candidates, artist_tracks], ignore_index=True)

        all_candidates.append(candidates)

    if all_candidates:
        return pd.concat(all_candidates, ignore_index=True)

    return pd.DataFrame(columns=["user_id", "track_id", "interaction_strength", "similarity_score", "source"])

