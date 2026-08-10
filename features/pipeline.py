from features.interaction_builder import build_interaction_dataframe
from features.embedding_builder import build_track_embeddings
from tabulate import tabulate


def run_features_pipeline(recent_tracks_df, top_tracks_df, tracks_df, model_name):
    try:
        interaction_df = build_interaction_dataframe(recent_tracks_df, top_tracks_df, tracks_df)
        track_embeddings = build_track_embeddings(tracks_df, model_name)

        print(f"\nInteraction DataFrame\n")
        print(tabulate(interaction_df.head(10), headers="keys", tablefmt="fancy_grid", showindex=False))
        print(f"\nTrack embeddings\n")
        for track_id, embedding in list(track_embeddings.items())[:10]:
            print(track_id, embedding.shape)

        return interaction_df, track_embeddings

    except Exception as e:
        print("\n Pipeline failed")
        print(f"Error: {e}")
        raise


