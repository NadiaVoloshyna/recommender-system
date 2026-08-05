from etl.pipeline import run_etl_pipeline
# from features.pipeline import build_features
# from vector_db.load_vectors import load_vectors
# from training.dataset_builder import build_training_dataset


def main():
    etl_data = run_etl_pipeline(fetch_api_data=False)

    # features = build_features(etl_data)

    # Store embeddings

    # Build ML dataset
    # training_data = build_training_dataset(
    #     interactions=features["interactions"],
    #     similarities=etl_data["track_similarity_df"],
    #     embeddings=features["embeddings"]
    # )


if __name__ == "__main__":
    main()

