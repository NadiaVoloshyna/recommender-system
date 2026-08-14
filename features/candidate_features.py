import pandas as pd
import numpy as np


def build_candidate_features(candidates: pd.DataFrame, source_overlap) -> pd.DataFrame:
    """
    Build ranking features from generated recommendation candidates.
    :param candidates:
    :param source_overlap:
    :return:
    """
    feature_df = candidates.copy()

    feature_df["track_signal"] = (
            feature_df["interaction_strength"].fillna(0)
            * feature_df["track_similarity_score"].fillna(0)
    )

    feature_df["artist_signal"] = (
            feature_df["interaction_strength"].fillna(0)
            * feature_df["artist_similarity_score"].fillna(0)
    )

    feature_df["vector_signal"] = (
        feature_df["vector_similarity_score"].fillna(0)
    )

    # Add source indicators
    feature_df["is_track_similarity"] = (feature_df["source"] == "track_similarity").astype(int)

    feature_df["is_artist_similarity"] = (feature_df["source"] == "artist_similarity").astype(int)

    feature_df["is_vector_similarity"] = (feature_df["source"] == "vector_similarity").astype(int)

    # Add number of sources per candidate
    feature_df = feature_df.merge(
        source_overlap.rename("n_sources").reset_index(),
        on=["user_id", "track_id"],
        how="left"
    )

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

