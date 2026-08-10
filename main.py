from etl.pipeline import run_etl_pipeline
from features.pipeline import run_features_pipeline
from vector_store.pipeline import run_vector_store_pipeline


def main():
    # Load and prepare data
    etl_data = run_etl_pipeline(fetch_api_data=False)
    recent_tracks_df = etl_data["recent_tracks_df"]
    top_tracks_df = etl_data["top_tracks_df"]
    top_artists_df = etl_data["top_artists_df"]
    users_df = etl_data["users_df"]
    artists_df = etl_data["artists_df"]
    tracks_df = etl_data["tracks_df"]
    track_similarity_df = etl_data["track_similarity_df"]
    artist_similarity_df = etl_data["artist_similarity_df"]

    # Build user-track interaction dataset. Create track embeddings
    interaction_df, track_embeddings = run_features_pipeline(
        recent_tracks_df,
        top_tracks_df,
        tracks_df,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Build FAISS vector index
    faiss_index, track_id_mapping = run_vector_store_pipeline(track_embeddings)


    # # Generate candidates for users
    # user_candidates = {}
    #
    # for user_id in interaction_df["user_id"].unique():
    #     # Candidates from user's listening history
    #     history_candidates = get_history_candidates(user_id, interaction_df)
    #
    #     # Candidates from external similarity API
    #     api_candidates = get_api_similarity_candidates(user_id, history_candidates)
    #
    #     # Candidates from FAISS vector similarity
    #     vector_candidates = get_vector_similarity_candidates(
    #         history_candidates,
    #         faiss_index,
    #         track_id_mapping,
    #         track_embeddings
    #     )
    #
    #     # Merge all candidate sources
    #     candidates = merge_candidates(
    #         history_candidates,
    #         api_candidates,
    #         vector_candidates
    #     )
    #
    #     user_candidates[user_id] = candidates
    #
    # # Build ranking dataset
    # training_data = build_training_dataset(
    #     interactions=interaction_df,
    #     candidates=user_candidates,
    #     embeddings=track_embeddings
    # )
    #
    # # Train ranking model (future step)
    # # model = train_ranker(training_data)


if __name__ == "__main__":
    main()


# def main():
#
#     features = create_features()
#
#     candidates = generate_candidates(
#         features
#     )
#
#     recommendations = rank_candidates(
#         candidates
#     )
#
#     return recommendations
