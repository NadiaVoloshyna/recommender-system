from etl.pipeline import run_etl_pipeline
from features.pipeline import run_features_pipeline
from features.candidate_features import build_candidate_features
from vector_store.pipeline import run_vector_store_pipeline
from candidates.generate_candidates import generate_candidates
from tabulate import tabulate


def main():
    # ETL
    etl_data = run_etl_pipeline(fetch_api_data=False)
    recent_tracks_df = etl_data["recent_tracks_df"]
    top_tracks_df = etl_data["top_tracks_df"]
    tracks_df = etl_data["tracks_df"]
    track_similarity_df = etl_data["track_similarity_df"]
    artist_similarity_df = etl_data["artist_similarity_df"]

    # User/item features
    interaction_df, track_embeddings = run_features_pipeline(
        recent_tracks_df,
        top_tracks_df,
        tracks_df,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector retrieval infrastructure
    faiss_index, track_id_mapping = run_vector_store_pipeline(track_embeddings)

    # Candidate generation
    candidates = generate_candidates(
        tracks_df=tracks_df,
        track_similarity_df=track_similarity_df,
        artist_similarity_df=artist_similarity_df,
        interaction_df=interaction_df,
        faiss_index=faiss_index,
        track_id_mapping=track_id_mapping,
        track_embeddings=track_embeddings,
        k_track_candidates=20,
        k_artist_candidates=10,
        k_artists=20,
        k_vector_candidates=20,
        similarity_threshold=0.70
    )

    print("\nCandidates\n")
    print(tabulate(candidates.head(10), headers="keys", tablefmt="fancy_grid", showindex=False))
    print("\nCandidates by source:")
    print(candidates["source"].value_counts())
    print("\nCandidates per user:")
    print(candidates["user_id"].value_counts())
    source_overlap = (
        candidates
        .groupby(["user_id", "track_id"])["source"]
        .nunique()
    )
    print("\nNumber of sources per candidate:")
    print(source_overlap.value_counts())

    # Candidate feature engineering
    feature_df = build_candidate_features(candidates=candidates, source_overlap=source_overlap)

    print("\nFeature_df\n")
    print(tabulate(feature_df.head(10), headers="keys", tablefmt="fancy_grid", showindex=False))
    print(feature_df.shape)
    print(feature_df.isna().mean())
    print(feature_df.describe())

    # Ranking
    # ranked_candidates = rank_candidates(feature_df)

    # Evaluation


if __name__ == "__main__":
    main()

