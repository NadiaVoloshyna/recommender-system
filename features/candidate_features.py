import pandas as pd
import numpy as np


def build_candidate_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Build ranking features from generated recommendation candidates.
    :param candidates:
    :return:
    """
    feature_df = candidates.copy()

    # Similarity signals
    feature_df["track_signal"] = (
            feature_df["interaction_strength"].fillna(0)
            * feature_df["track_similarity_score"].fillna(0)
    )

    feature_df["artist_signal"] = (
            feature_df["interaction_strength"].fillna(0)
            * feature_df["artist_similarity_score"].fillna(0)
    )

    feature_df["vector_signal"] = (
        feature_df["interaction_strength"].fillna(0)
        * feature_df["vector_similarity_score"].fillna(0)
    )

    # Number of different sources that generated this candidate
    source_overlap = (
        candidates.groupby(["user_id", "track_id"])["source"]
        .nunique()
        .rename("n_sources")
        .reset_index()
    )

    feature_df = feature_df.merge(
        source_overlap,
        on=["user_id", "track_id"],
        how="left",
    )

    feature_df = (feature_df.groupby(["user_id", "track_id"], as_index=False).agg({
            "interaction_strength": "max",
            "track_similarity_score": "max",
            "artist_similarity_score": "max",
            "vector_similarity_score": "max",
            "track_signal": "max",
            "artist_signal": "max",
            "vector_signal": "max",
            "n_sources": "max"
        })
    )

    feature_df["track_similarity_score"] = (feature_df["track_similarity_score"].fillna(0))
    feature_df["artist_similarity_score"] = (feature_df["artist_similarity_score"].fillna(0))
    feature_df["vector_similarity_score"] = (feature_df["vector_similarity_score"].fillna(0))
    feature_df["interaction_strength"] = (feature_df["interaction_strength"].fillna(0))

    # Normalize interaction_strength per user
    # feature_df["interaction_strength_log"] = np.log1p(
    #     feature_df["interaction_strength"]
    # )
    #
    # user_max = feature_df.groupby("user_id")["interaction_strength"].transform("max")
    #
    # feature_df["interaction_strength_normalized"] = (
    #         feature_df["interaction_strength"] / user_max
    # )

    return feature_df

