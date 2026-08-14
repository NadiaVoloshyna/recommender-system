import pandas as pd
from features.utils import validate_columns
import numpy as np


def get_history_tracks(user_id: str, interaction_df: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieve a user's listening history from the interaction data.
    Filters the interaction DataFrame to include only records associated with the specified user and returns
    the user's track IDs along with their interaction strengths.
    :param user_id: the ID of the user whose listening history is being retrieved (str)
    :param interaction_df: contains user-track interactions. It must include the columns:
    user_id, track_id, and interaction_strength (pd.DataFrame)
    :return: DataFrame containing the columns: user_id, track_id, and interaction_strength for the specified user.
    """
    history_tracks = interaction_df[interaction_df["user_id"] == user_id][
        ["user_id", "track_id", "interaction_strength"]].copy()

    return history_tracks


def get_similar_track_candidates(
        user_id: str,
        history_tracks: pd.DataFrame,
        track_similarity_df: pd.DataFrame,
        k_sim: int,
        similarity_threshold: float
) -> pd.DataFrame:
    """
    Generate track recommendation candidates based on track similarity.
    For each track in the user's listening history, finds up to k_sim unique tracks whose similarity score meets
    or exceeds the specified threshold. Candidates are ordered by descending similarity score.
    The source history track and its interaction strength are carried over to each candidate.
    The track similarity score is stored in track_similarity_score, while artist_similarity_score is set to
    None because these candidates are generated using track similarity.
    :param user_id: ID of the user whose listening history is used to generate recommendations (str)
    :param history_tracks: contains the user's listening history. It must include the columns
    track_id and interaction_strength (pd.DataFrame)
    :param track_similarity_df: contains track similarity information. It must include the columns
    track_id, similar_track_id, and similarity_score (pd.DataFrame)
    :param k_sim: maximum number of unique similar tracks to return for each track in the user's listening history.
    Must be greater than zero (int)
    :param similarity_threshold: minimum similarity score required for a track to be considered a candidate.
    Must be between 0 and 1 inclusive (float)
    :return: a DataFrame containing the columns: user_id, track_id, source_track_id, interaction_strength,
    track_similarity_score, artist_similarity_score, and source.
    track_id represents the recommended similar track
    source_track_id identifies the history track that generated the candidate
    interaction_strength is inherited from the source history track
    track_similarity_score represents the similarity between the source track and the recommended track
    artist_similarity_score is always None because candidates are generated using track similarity
    source is always "track_similarity" for these candidates
    """
    validate_columns(history_tracks, ["track_id", "interaction_strength"], "history_candidates")
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

    # Tracks already listened to by the user should not be recommended.
    history_track_ids = set(history_tracks["track_id"])

    for _, history_row in history_tracks.iterrows():
        track_id = history_row["track_id"]
        interaction_strength = history_row["interaction_strength"]
        sims = (
            track_similarity_df[
                (track_similarity_df["track_id"] == track_id)
                & (track_similarity_df["similarity_score"] >= similarity_threshold)
                ]
            .sort_values("similarity_score", ascending=False)
            .drop_duplicates("similar_track_id")
            .copy()
        )

        # Remove tracks already present in the user's history
        sims = sims[~sims["similar_track_id"].isin(history_track_ids)]

        # Keep the top k_sim after filtering.
        sims = sims.head(k_sim)

        if sims.empty:
            continue

        sims = sims[["similar_track_id", "similarity_score"]].rename(
            columns={"similar_track_id": "track_id", "similarity_score": "track_similarity_score"})
        sims["user_id"] = user_id
        sims["source_track_id"] = track_id
        sims["interaction_strength"] = interaction_strength
        sims["artist_similarity_score"] = None
        sims["vector_similarity_score"] = None
        sims["source"] = "track_similarity"

        candidates.append(sims[[
            "user_id", "track_id", "source_track_id", "interaction_strength", "track_similarity_score",
            "artist_similarity_score", "vector_similarity_score", "source"
        ]])

    if not candidates:
        return pd.DataFrame(
            columns=["user_id", "track_id", "source_track_id", "interaction_strength", "track_similarity_score",
                     "artist_similarity_score", "vector_similarity_score", "source"])

    return pd.concat(candidates, ignore_index=True).drop_duplicates().reset_index(drop=True)


def get_similar_artist_candidates(
        user_id: str,
        history_tracks: pd.DataFrame,
        artist_similarity_df: pd.DataFrame,
        tracks_df: pd.DataFrame,
        k_sim: int,
        k_artists: int,
        similarity_threshold: float
) -> pd.DataFrame:
    """
    Generate track recommendation candidates based on artist similarity.
    For each track in the user's listening history, identifies its source artist and finds up to k_artists
    similar artists whose similarity score meets or exceeds similarity_threshold. Tracks belonging to
    those similar artists are then considered as recommendation candidates.
    Tracks already present in the user's listening history are excluded. The resulting candidates are ranked
    by artist similarity score, and at most k_candidates tracks are retained for each source history track.
    The source history track and its interaction strength are carried over to each candidate. The artist similarity
    score is stored in artist_similarity_score, while track_similarity_score is set to None
    because these candidates are generated using artist similarity.
    :param user_id: ID of the user whose listening history is used to generate recommendations (str)
    :param history_tracks: contains the user's listening history. It must include the columns
    track_id, source_track_id, and interaction_strength (pd.DataFrame)
    :param artist_similarity_df: contains artist similarity information. It must include the columns:
    artist_id, similar_artist_id, and similarity_score (pd.DataFrame)
    :param tracks_df: a DataFrame mapping tracks to their artists. It must include the columns:
    track_id and artist_id (pd.DataFrame)
    :param k_sim: maximum number of recommended tracks to return for each history track.
    Must be greater than zero (int)
    :param k_artists: maximum number of similar artists to consider for each source artist.
    Must be greater than zero (int)
    :param similarity_threshold: minimum artist similarity score required for an artist to be considered
    as a candidate. Must be between 0 and 1 inclusive (float)
    :return: a DataFrame containing the columns:
        user_id`: ID of the user
        track_id: recommended track belonging to a similar artist
        source_track_id: history track whose artist similarity generated the candidate
        interaction_strength: interaction strength inherited from the source history track
        track_similarity_score: always None because candidates are generated using artist
        similarity rather than track similarity
        artist_similarity_score: similarity score between the source artist and the similar artist
        source: always "artist_similarity" for these candidates.
    """

    validate_columns(history_tracks, ["track_id", "interaction_strength"], "history_tracks")
    validate_columns(artist_similarity_df, ["artist_id", "similar_artist_id", "similarity_score"], "artist_similarity_df")
    validate_columns(tracks_df, ["track_id", "artist_id"], "tracks_df")

    if not isinstance(k_sim, int) or isinstance(k_sim, bool):
        raise TypeError("k_sim must be an integer")

    if k_sim <= 0:
        raise ValueError("k_sim must be greater than 0")

    if not isinstance(k_artists, int) or isinstance(k_artists, bool):
        raise TypeError("k_artists must be an integer")

    if k_artists <= 0:
        raise ValueError("k_artists must be greater than 0")

    if not isinstance(similarity_threshold, (int, float)) or isinstance(similarity_threshold, bool):
        raise TypeError("similarity_threshold must be numeric")

    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1")

    candidates = []

    # Connect each history track to its artist
    history_artists = (
        history_tracks[["track_id", "interaction_strength"]]
        .merge(
            tracks_df[["track_id", "artist_id"]],
            on="track_id",
            how="left"
        )
    )
    # Tracks already listened to by the user should not be recommended.
    history_track_ids = set(history_tracks["track_id"])

    for _, history_row in history_artists.iterrows():
        source_artist_id = history_row["artist_id"]
        source_track_id = history_row["track_id"]
        interaction_strength = history_row["interaction_strength"]

        # Find similar artists above the similarity threshold
        similar_artists = (
            artist_similarity_df[(artist_similarity_df["artist_id"] == source_artist_id)
                                 & (artist_similarity_df["similarity_score"] >= similarity_threshold)]
            .sort_values("similarity_score", ascending=False)
            .drop_duplicates("similar_artist_id")
            .head(k_artists)
            .copy()
        )

        if similar_artists.empty:
            continue

        # Connect similar artists to their tracks
        artist_tracks = tracks_df[["artist_id", "track_id"]].merge(
            similar_artists[["similar_artist_id", "similarity_score"]],
            left_on="artist_id",
            right_on="similar_artist_id",
            how="inner"
        )

        # Remove tracks already present in the user's history
        artist_tracks = artist_tracks[~artist_tracks["track_id"].isin(history_track_ids)].copy()

        # Remove duplicate track recommendations, keeping the highest artist similarity score
        # if a track appears more than once
        artist_tracks = (
            artist_tracks
            .sort_values("similarity_score", ascending=False)
            .drop_duplicates("track_id")
            .head(k_sim)
        )

        if artist_tracks.empty:
            continue

        # Add recommendation metadata
        artist_tracks["user_id"] = user_id
        artist_tracks["source_track_id"] = source_track_id
        artist_tracks["interaction_strength"] = interaction_strength
        artist_tracks["track_similarity_score"] = None
        artist_tracks["artist_similarity_score"] = (artist_tracks["similarity_score"])
        artist_tracks["vector_similarity_score"] = None
        artist_tracks["source"] = "artist_similarity"

        candidates.append(
            artist_tracks[[
                "user_id", "track_id", "source_track_id", "interaction_strength",  "track_similarity_score",
                "artist_similarity_score", "vector_similarity_score", "source"
                ]])

        if not candidates:
            return pd.DataFrame(
                columns=["user_id", "track_id", "source_track_id", "interaction_strength", "track_similarity_score",
                         "artist_similarity_score", "vector_similarity_score", "source"])

    return pd.concat(candidates, ignore_index=True).drop_duplicates().reset_index(drop=True)


def get_vector_candidates(
        user_id: str,
        history_tracks: pd.DataFrame,
        faiss_index,
        track_id_mapping: list[str],
        track_embeddings: dict,
        k: int
) -> pd.DataFrame:
    """
    Generate track recommendation candidates using vector similarity.
    Creates a user profile vector by averaging the embeddings of tracks in the user's listening history
    and retrieves the k-nearest tracks from the FAISS index.
    Vector candidates do not have a single source history track because the user profile vector is constructed
    from the user's entire history. Therefore, source_track_id and interaction_strength are set to None.
    Already-listened tracks are excluded from the candidates.
    :param user_id: ID of the user whose recommendations are being generated (str)
    :param history_tracks: DataFrame containing the user's listening history.
        Must include a track_id column (pd.DataFrame)
    :param faiss_index: loaded FAISS index containing the track embeddings
    :param track_id_mapping: list mapping FAISS index positions to track IDs (list[str])
    :param track_embeddings: mapping from track IDs to their embedding vectors (dict)
    :param k: maximum number of vector-based candidates to retrieve (int)
    :return: DataFrame containing the columns: user_id, track_id, source_track_id, interaction_strength,
    track_similarity_score, artist_similarity_score, vector_similarity_score, source
    source_track_id is NaN
    interaction_strength is NaN
    track_similarity_score is NaN
    artist_similarity_score is NaN
    vector_similarity_score contains the FAISS similarity score
    source is "vector_similarity"
    """
    output_columns = ["user_id", "track_id", "source_track_id", "interaction_strength", "track_similarity_score",
                      "artist_similarity_score", "vector_similarity_score", "source"]

    validate_columns(history_tracks, ["track_id"], "history_tracks")

    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("k must be an integer")

    if k <= 0:
        raise ValueError("k must be greater than 0")

    # Cold-start user
    if history_tracks.empty:
        return pd.DataFrame(columns=output_columns)

    #  Get embeddings for tracks in the user's history
    embeddings = []

    for track_id in history_tracks["track_id"]:
        embedding = track_embeddings.get(track_id)
        if embedding is not None:
            embeddings.append(embedding)

    # No embeddings available for this user's history
    if not embeddings:
        return pd.DataFrame(columns=output_columns)

    # Create user profile vector
    user_vector = np.mean(np.asarray(embeddings), axis=0).astype(np.float32)

    # Tracks already listened to by the user
    history_track_ids = set(history_tracks["track_id"])

    # Retrieve additional tracks to compensate for tracks that will be removed because they are already in history.
    search_k = min(k + len(history_track_ids), faiss_index.ntotal)

    if search_k == 0:
        return pd.DataFrame(columns=output_columns)

    # Retrieve k-nearest tracks
    distances, indices = faiss_index.search(user_vector.reshape(1, -1), k=search_k)

    candidates = []

    for similarity_score, index in zip(distances[0], indices[0]):
        # Ignore invalid FAISS indices
        if index == -1:
            continue

        track_id = track_id_mapping[index]

        # Do not recommend tracks already in history
        if track_id in history_track_ids:
            continue

        candidates.append({
            "user_id": user_id,
            "track_id": track_id,
            "source_track_id": np.nan,
            "interaction_strength": np.nan,
            "track_similarity_score": np.nan,
            "artist_similarity_score": np.nan,
            "vector_similarity_score": float(similarity_score),
            "source": "vector_similarity"
        })

        # Keep at most k recommendation candidates
        if len(candidates) >= k:
            break

    if not candidates:
        return pd.DataFrame(columns=output_columns)

    return (
        pd.DataFrame(candidates, columns=output_columns)
        .drop_duplicates()
        .reset_index(drop=True)
    )


def generate_candidates(
        tracks_df,
        track_similarity_df,
        artist_similarity_df,
        interaction_df,
        faiss_index,
        track_id_mapping,
        track_embeddings,
        k_track_candidates,
        k_artist_candidates,
        k_artists,
        k_vector_candidates,
        similarity_threshold
):
    all_candidates = []

    for user_id in interaction_df["user_id"].unique():
        history_tracks = get_history_tracks(
            user_id,
            interaction_df
        )

        track_candidates = get_similar_track_candidates(
            user_id,
            history_tracks,
            track_similarity_df,
            k_sim=k_track_candidates,
            similarity_threshold=similarity_threshold
        )

        artist_tracks = get_similar_artist_candidates(
            user_id,
            history_tracks,
            artist_similarity_df,
            tracks_df,
            k_sim=k_artist_candidates,
            k_artists=k_artists,
            similarity_threshold=similarity_threshold
        )

        vector_candidates = get_vector_candidates(
            user_id,
            history_tracks,
            faiss_index,
            track_id_mapping,
            track_embeddings,
            k=k_vector_candidates
        )

        candidates = pd.concat([track_candidates, artist_tracks, vector_candidates], ignore_index=True)

        all_candidates.append(candidates)

    if all_candidates:
        return pd.concat(all_candidates, ignore_index=True)

    return pd.DataFrame(columns=["user_id", "track_id", "source_track_id", "interaction_strength",
                                 "track_similarity_score", "artist_similarity_score",
                                 "vector_similarity_score", "source"])

