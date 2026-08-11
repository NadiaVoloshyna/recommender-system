import pandas as pd
from features.utils import validate_columns


def get_history_tracks(user_id: str, interaction_df: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieve and format a user's listening history.
    Filters the interaction DataFrame to find all tracks associated with the specified user, removes duplicate rows,
    and keeps the user's interaction strength for each track. Adds a similarity_score column with a value of None
    and a source column labelled "history".
    :param user_id: the ID of the user whose interaction history is being retrieved (str)
    :param interaction_df: contains user-track interactions, including the columns user_id, track_id,
    and interaction_strength (pd.DataFrame)
    :return: DataFrame containing user_id, track_id, interaction_strength, similarity_score, and source columns.
    The source column identifies these tracks as coming from the user's interaction history.
    """
    history_tracks = interaction_df[interaction_df["user_id"] == user_id][
        ["user_id", "track_id", "interaction_strength"]]

    history_tracks = history_tracks.assign(similarity_score=None, source="history")

    return history_tracks


def get_similar_track_candidates(
        user_id: str,
        history_candidates: pd.DataFrame,
        track_similarity_df: pd.DataFrame,
        k_sim: int,
        similarity_threshold: float
) -> pd.DataFrame:
    """
    Finds similar tracks to those in a user's listening history.
    Takes the tracks from a user's history and finds the top k_sim sufficiently similar tracks for each one.
    Filters out tracks below the similarity threshold, orders the remaining tracks by similarity score,
    and formats them as recommendation candidates.
    :param user_id: ID of the user whose interaction history is being used to generate recommendations (str)
    :param history_candidates: a DataFrame containing the user's interaction history, including a track_id column
    :param track_similarity_df: a DataFrame containing the original track_id, similar_track_id,
    and similarity_score for pairs of similar tracks
    :param k_sim: maximum number of similar tracks to return for each track in the user's history (int)
    :param similarity_threshold: minimum similarity score required for a track to be considered a candidate (float)
    :return: a DataFrame containing the columns: user_id, track_id, interaction_strength, similarity_score, and source.
    The track_id represents the similar track, interaction_strength is set to None,
    and source is set to "track_similarity".
    """
    validate_columns(history_candidates, ["track_id"], "history_candidates")

    validate_columns(track_similarity_df, ["track_id", "similar_track_id", "similarity_score"], "track_similarity_df")

    if not isinstance(k_sim, int) or isinstance(k_sim, bool):
        raise TypeError("k_sim must be an integer")

    if k_sim <= 0:
        raise ValueError("k_sim must be greater than 0")

    if not isinstance(similarity_threshold, (int, float)) or isinstance(similarity_threshold, bool):
        raise TypeError("similarity_threshold must be numeric")

    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1")

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

