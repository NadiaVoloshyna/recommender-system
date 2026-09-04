from etl.pipeline import run_etl_pipeline
from features.pipeline import run_features_pipeline
from features.candidate_features import build_candidate_features
from vector_store.pipeline import run_vector_store_pipeline
from candidates.generate_candidates import generate_candidates
from features.labels import split_user_history, split_training_history, add_labels
from features.analysis import analyze_candidates, analyze_features,analyze_labels, analyze_candidate_recall
from training.negative_sampling import sample_negatives
from pathlib import Path
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

    # Interactions outer split: 80/10/10
    train_interactions, val_interactions, test_interactions = split_user_history(
        interaction_df,
        val_size=0.1,
        test_size=0.1,
        random_state=42,
    )

    # Vector retrieval infrastructure
    faiss_index, track_id_mapping = run_vector_store_pipeline(track_embeddings)

    # ========== TRAINING CANDIDATES ==========

    # Training-period split: 80/20
    train_history, train_targets = split_training_history(
        train_interactions,
        target_size=0.2,
        random_state=42,
    )

    # Candidates generation
    train_candidates = generate_candidates(
        tracks_df=tracks_df,
        track_similarity_df=track_similarity_df,
        artist_similarity_df=artist_similarity_df,
        interaction_df=train_history,
        faiss_index=faiss_index,
        track_id_mapping=track_id_mapping,
        track_embeddings=track_embeddings,
        k_track_candidates=50,
        k_artist_candidates=100,
        k_artists=50,
        k_vector_candidates=2600
    )

    # Candidates feature engineering
    train_feature_df = build_candidate_features(candidates=train_candidates, interaction_df=train_history)

    # Add labels using held-out interactions
    train_feature_df = add_labels(train_feature_df, train_targets)

    # Analysis - FULL TRAINING CANDIDATE POOL
    print("\n========== Analysis - FULL TRAINING CANDIDATE POOL ==========")
    analyze_candidates(train_candidates)
    analyze_features(train_feature_df)
    analyze_labels(train_feature_df)
    analyze_candidate_recall(train_candidates, train_targets)

    # Training-data preparation
    training_df = sample_negatives(
        train_feature_df,
        negatives_per_positive=10,
        hard_ratio=0.50,
        medium_ratio=0.30,
        random_ratio=0.20,
        random_state=42,
    )

    # Persist training data
    feature_path = Path("artifacts/features/train_features_50_100_50_2600.parquet")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_parquet(feature_path, index=False)

    # ========== VALIDATION CANDIDATES ==========

    val_candidates = generate_candidates(
        tracks_df=tracks_df,
        track_similarity_df=track_similarity_df,
        artist_similarity_df=artist_similarity_df,
        interaction_df=train_interactions,
        faiss_index=faiss_index,
        track_id_mapping=track_id_mapping,
        track_embeddings=track_embeddings,
        k_track_candidates=50,
        k_artist_candidates=100,
        k_artists=50,
        k_vector_candidates=2600,
    )

    val_feature_df = build_candidate_features(
        candidates=val_candidates,
        interaction_df=train_interactions,
    )

    val_feature_df = add_labels(
        val_feature_df,
        val_interactions,
    )

    # Analysis - VALIDATION CANDIDATE POOL
    # print("\n========== Analysis - VALIDATION CANDIDATE POOL ==========")
    # analyze_candidates(val_candidates)
    # analyze_features(val_feature_df)
    # analyze_labels(val_feature_df)
    # analyze_candidate_recall(val_candidates, val_interactions)

    # Persist validation data
    feature_path = Path("artifacts/features/val_features_50_100_50_2600.parquet")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    val_feature_df.to_parquet(feature_path, index=False)

    # ========== TEST CANDIDATES ==========

    test_history = pd.concat(
        [train_interactions, val_interactions],
        ignore_index=True,
    )

    test_candidates = generate_candidates(
        tracks_df=tracks_df,
        track_similarity_df=track_similarity_df,
        artist_similarity_df=artist_similarity_df,
        interaction_df=test_history,
        faiss_index=faiss_index,
        track_id_mapping=track_id_mapping,
        track_embeddings=track_embeddings,
        k_track_candidates=50,
        k_artist_candidates=100,
        k_artists=50,
        k_vector_candidates=2600,
    )

    test_feature_df = build_candidate_features(
        candidates=test_candidates,
        interaction_df=test_history,
    )

    test_feature_df = add_labels(
        test_feature_df,
        test_interactions,
    )

    # Analysis - TEST CANDIDATE POOL
    # print("\n========== Analysis - TEST CANDIDATE POOL ==========")
    # analyze_candidates(test_candidates)
    # analyze_features(test_feature_df)
    # analyze_labels(test_feature_df)
    # analyze_candidate_recall(test_candidates, test_interactions)

    # Persist test data
    feature_path = Path("artifacts/features/test_features_50_100_50_2600.parquet")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    test_feature_df.to_parquet(feature_path, index=False)

    # Train ranker
    # model = train_model(training_df)

    # Validation/test evaluation
    # val_predictions = model.predict(val_feature_df)
    # test_predictions = model.predict(test_feature_df)


if __name__ == "__main__":
    main()
