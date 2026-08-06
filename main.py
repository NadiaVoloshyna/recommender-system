from etl.pipeline import run_etl_pipeline
from features.interaction_builder import build_interaction_dataframe
# from vector_db.load_vectors import load_vectors
# from training.dataset_builder import build_training_dataset
from tabulate import tabulate


def main():
    etl_data = run_etl_pipeline(fetch_api_data=False)
    recent_tracks_df = etl_data["recent_tracks_df"]
    top_tracks_df = etl_data["top_tracks_df"]
    tracks_df = etl_data["tracks_df"]

    interaction_df = build_interaction_dataframe(recent_tracks_df, top_tracks_df, tracks_df)
    print(f"\nInteraction DataFrame\n")
    print(tabulate(interaction_df.head(10), headers="keys", tablefmt="fancy_grid", showindex=False))

    # Store embeddings

    # Build ML dataset
    # training_data = build_training_dataset(
    #     interactions=features["interactions"],
    #     similarities=etl_data["track_similarity_df"],
    #     embeddings=features["embeddings"]
    # )


if __name__ == "__main__":
    main()

