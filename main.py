from etl.pipeline import run_etl_pipeline
from features.pipeline import run_features_pipeline
from features.candidate_features import build_candidate_features
from vector_store.pipeline import run_vector_store_pipeline
from candidates.generate_candidates import generate_candidates
from tabulate import tabulate
import pandas as pd


def main():
    # ETL
    etl_data = run_etl_pipeline(fetch_api_data=False)
    recent_tracks_df = etl_data["recent_tracks_df"]
    top_tracks_df = etl_data["top_tracks_df"]
    tracks_df = etl_data["tracks_df"]
    track_similarity_df = etl_data["track_similarity_df"]
    artist_similarity_df = etl_data["artist_similarity_df"]

    # User-item features
    interaction_df, track_embeddings = run_features_pipeline(
        recent_tracks_df,
        top_tracks_df,
        tracks_df,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector retrieval infrastructure
    faiss_index, track_id_mapping = run_vector_store_pipeline(track_embeddings)

    # Candidates generation
    candidates = generate_candidates(
        tracks_df=tracks_df,
        track_similarity_df=track_similarity_df,
        artist_similarity_df=artist_similarity_df,
        interaction_df=interaction_df,
        faiss_index=faiss_index,
        track_id_mapping=track_id_mapping,
        track_embeddings=track_embeddings,
        k_track_candidates=50,
        k_artist_candidates=30,
        k_artists=20,
        k_vector_candidates=50
    )

    # Candidates analysis
    print("\nPreview candidate rows\n")
    display_candidates = candidates.head(10).copy()
    for col in ["user_id", "track_id", "source_track_id"]:
        display_candidates[col] = display_candidates[col].astype(str).str[:8] + "..."
    print(tabulate(display_candidates, headers="keys", tablefmt="fancy_grid", showindex=False))

    print("\nCandidate counts by retrieval source:")
    print(candidates["source"].value_counts())

    print("\nNumber of candidates per user:")
    print(candidates["user_id"].value_counts())

    source_overlap = (candidates.groupby(["user_id", "track_id"])["source"].nunique())
    print("\nNumber of sources per candidate:")
    print(source_overlap.value_counts().sort_index())

    print("\nUnique candidates per user and source:")
    print(candidates.groupby(["user_id", "source"])["track_id"].nunique().groupby("source").describe())

    print("\nSimilarity score distributions by source:")
    for source in candidates["source"].unique():
        print(f"\n{source}")
        source_df = candidates[candidates["source"] == source]
        score_column = {
            "track_similarity": "track_similarity_score",
            "artist_similarity": "artist_similarity_score",
            "vector_similarity": "vector_similarity_score",
        }[source]
        print(source_df[score_column].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

    # Candidates feature engineering
    feature_df = build_candidate_features(candidates=candidates, interaction_df=interaction_df)

    # Features analysis
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    print("\nFeature_df\n")
    display_feature_df = feature_df.head(10).copy()
    for col in ["user_id", "track_id"]:
        display_feature_df[col] = display_feature_df[col].astype(str).str[:8] + "..."
    print(tabulate(display_feature_df, headers="keys", tablefmt="fancy_grid", showindex=False))

    print("\nShape:")
    print(feature_df.shape)

    print("\nFeature distributions:")
    distribution_columns = [
        "source_interaction_strength",

        "track_similarity_score",
        "artist_similarity_score",
        "vector_similarity_score",

        "n_sources",

        "global_popularity_log",
        "global_popularity_missing",
        "candidate_relative_global_popularity",

        "max_similarity",
        "mean_similarity",

        "track_signal",
        "artist_signal",
        "vector_signal"]
    print(feature_df[distribution_columns].describe())

    print("\nSimilarity feature coverage:")
    for col in ["track_similarity_score", "artist_similarity_score", "vector_similarity_score"]:
        print(f"{col}: {(feature_df[col] > 0).mean():.1%}")

    print("\nCandidates by number of retrieval sources:")
    print(feature_df["n_sources"].value_counts().sort_index())

    print("\nGlobal popularity coverage:")
    print(
        f"Popularity available: "
        f"{(feature_df['global_popularity_missing'] == 0).mean():.1%}")
    print(
        f"Global popularity missing: "
        f"{(feature_df['global_popularity_missing'] == 1).mean():.1%}")

    print("\nGlobal popularity distribution:")
    print(
        feature_df["global_popularity_log"].describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

    print("\nCandidate-relative popularity:")
    print(feature_df["candidate_relative_global_popularity"].describe())

    print("\nZero signal proportions:")
    for col in ["track_signal", "artist_signal", "vector_signal"]:
        print(f"{col}: {(feature_df[col] == 0).mean():.1%}")

    print("\nSignal ranges:")
    for col in ["track_signal", "artist_signal", "vector_signal"]:
        print(f"{col}: "f"min={feature_df[col].min():.3f}, "f"max={feature_df[col].max():.3f}")

    # Ranking
    # ranked_candidates = rank_candidates(feature_df)


if __name__ == "__main__":
    main()
