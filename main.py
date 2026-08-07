from etl.pipeline import run_etl_pipeline
from features.interaction_builder import build_interaction_dataframe
from features.embedding_builder import build_track_embeddings
from vector_store.faiss_builder import build_faiss
from tabulate import tabulate


def main():
    # Load and prepare data
    etl_data = run_etl_pipeline(fetch_api_data=False)
    recent_tracks_df = etl_data["recent_tracks_df"]
    top_tracks_df = etl_data["top_tracks_df"]
    tracks_df = etl_data["tracks_df"]

    # Build user-track interaction dataset. Used for user history candidates and ML features
    interaction_df = build_interaction_dataframe(recent_tracks_df, top_tracks_df, tracks_df)
    print(f"\nInteraction DataFrame\n")
    print(tabulate(interaction_df.head(10), headers="keys", tablefmt="fancy_grid", showindex=False))

    # Create track embeddings. Used for vector similarity search
    track_embeddings = build_track_embeddings(tracks_df)
    print(f"\nTrack embeddings\n")
    for track_id, embedding in list(track_embeddings.items())[:10]:
        print(track_id, embedding.shape)

    # Build FAISS vector index. Used to retrieve semantically similar tracks
    faiss_index, track_id_mapping = build_faiss(track_embeddings)
    print("\nFAISS Index\n")
    print(f"Number of vectors: {faiss_index.ntotal}")
    print(f"Embedding dimension: {faiss_index.d}")
    print(f"Number of track IDs: {len(track_id_mapping)}")

    print("\nFAISS ID mapping sample\n")
    for i, track_id in enumerate(track_id_mapping[:10]):
        print(i, track_id)

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
