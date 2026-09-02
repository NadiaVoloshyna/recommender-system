from etl.pipeline import run_etl_pipeline
from features.pipeline import run_features_pipeline
from features.candidate_features import build_candidate_features
from vector_store.pipeline import run_vector_store_pipeline
from candidates.generate_candidates import generate_candidates
from features.labels import split_user_history, add_labels
from features.analysis import analyze_candidates, analyze_features,analyze_labels, analyze_candidate_recall
from pathlib import Path


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

    # Split interactions
    train_part, held_out = split_user_history(
        interaction_df,
        test_size=0.2,
        random_state=42,
    )

    # Vector retrieval infrastructure
    faiss_index, track_id_mapping = run_vector_store_pipeline(track_embeddings)

    # Candidates generation
    candidates = generate_candidates(
        tracks_df=tracks_df,
        track_similarity_df=track_similarity_df,
        artist_similarity_df=artist_similarity_df,
        interaction_df=train_part,
        faiss_index=faiss_index,
        track_id_mapping=track_id_mapping,
        track_embeddings=track_embeddings,
        k_track_candidates=60,
        k_artist_candidates=30,
        k_artists=25,
        k_vector_candidates=300
    )

    # Candidates feature engineering
    feature_df = build_candidate_features(candidates=candidates, interaction_df=train_part)

    # Add labels using held-out interactions
    feature_df = add_labels(feature_df, held_out)

    # Persist features
    feature_path = Path("artifacts/features/60_30_25_300.parquet")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(feature_path, index=False)

    # Analysis
    analyze_candidates(candidates)
    analyze_features(feature_df)
    analyze_labels(feature_df)
    analyze_candidate_recall(candidates, held_out)

    # Create training set
    # feature_df = pd.read_parquet("data/features/60_30_25_300.parquet")
    # training_df = sample_negatives(
    #     feature_df,
    #     negatives_per_positive=10,
    #     random_state=42
    # )

    # Train
    # model = train_model(training_df)

    # Rank candidates using trained model
    # ranked_candidates = rank_candidates(feature_df, model)


if __name__ == "__main__":
    main()
