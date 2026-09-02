from tabulate import tabulate
import pandas as pd


def analyze_candidates(candidates):
    print("\n=== Candidate Analysis ===")

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


def analyze_features(feature_df):
    print("\n=== Feature Analysis ===")

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
        # Interaction
        "source_interaction_strength",
        "source_interaction_strength_log",

        # Candidate support
        "n_sources",

        # Similarity
        "track_similarity_score",
        "artist_similarity_score",
        "vector_similarity_score",
        "max_similarity",
        "mean_similarity_available",
        "mean_similarity_all",

        # Interaction × similarity
        "track_interaction_signal",
        "artist_interaction_signal",
        "vector_interaction_signal",

        # Popularity
        "global_popularity",
        "global_popularity_log",
        "candidate_relative_global_popularity",
        "global_popularity_missing",

        # Availability
        "track_similarity_available",
        "artist_similarity_available",
        "vector_similarity_available"
    ]
    print(feature_df[distribution_columns].describe())

    print("\nSimilarity feature coverage:")
    for col in ["track_similarity_score", "artist_similarity_score", "vector_similarity_score"]:
        print(f"{col}: {(feature_df[col] > 0).mean():.1%}")

    print("\nCandidates by number of retrieval sources:")
    print(feature_df["n_sources"].value_counts().sort_index())
    print(f"Candidates from multiple sources: "f"{(feature_df['n_sources'] > 1).mean():.1%}")

    print("\nGlobal popularity coverage:")
    popularity_available = (feature_df["global_popularity_missing"] == 0).mean()
    popularity_missing = (feature_df["global_popularity_missing"] == 1).mean()
    print(f"Popularity available: {popularity_available:.1%}")
    print(f"Global popularity missing: {popularity_missing:.1%}")

    print("\nZero interaction × similarity signal proportions:")
    for col in ["track_interaction_signal", "artist_interaction_signal", "vector_interaction_signal"]:
        print(f"{col}: {(feature_df[col] == 0).mean():.1%}")

    print("\nInteraction × similarity signal ranges:")
    for col in ["track_interaction_signal", "artist_interaction_signal", "vector_interaction_signal"]:
        print(f"{col}: "f"min={feature_df[col].min():.3f}, "f"max={feature_df[col].max():.3f}")

    print("\nInteraction × similarity signals when similarity is available:")
    signal_pairs = [("track_interaction_signal", "track_similarity_score"),
                    ("artist_interaction_signal", "artist_similarity_score"),
                    ("vector_interaction_signal", "vector_similarity_score")]
    for signal_col, similarity_col in signal_pairs:
        available = feature_df[similarity_col] > 0
        print(f"\n{signal_col}:")
        print(feature_df.loc[available, signal_col].describe())


def analyze_labels(feature_df):
    print("\n=== Label Analysis ===")

    print("\nLabel distribution:")
    print(
        feature_df["label"]
        .value_counts()
        .sort_index()
        .rename(index={0: "Negative", 1: "Positive"})
    )

    print("\nLabel proportions:")
    print(
        feature_df["label"]
        .value_counts(normalize=True)
        .sort_index()
        .rename(index={0: "Negative", 1: "Positive"})
        .map(lambda x: f"{x:.2%}")
    )

    print("\nPositive examples per user:")
    positive_per_user = (feature_df.groupby("user_id")["label"].sum())
    print(positive_per_user.describe())

    print("\nUsers with at least one positive candidate:")
    users_with_positive = (feature_df.groupby("user_id")["label"].max().mean())
    print(f"{users_with_positive:.2%}")

    print("\nCandidates per user by label:")
    candidates_by_label = (
        feature_df.groupby(["user_id", "label"])
        .size()
        .unstack(fill_value=0)
    )
    print(candidates_by_label.describe())

    print("\nFeature distributions by label:")
    print(feature_df.groupby("label").mean(numeric_only=True).T)


def analyze_candidate_recall(candidates, held_out):
    print("\n=== Candidate Recall Analysis ===")
    print("\nCandidate recall by retrieval source:")
    held_out_pairs = set(zip(held_out["user_id"], held_out["track_id"]))
    for source in candidates["source"].unique():
        source_pairs = set(
            zip(
                candidates.loc[candidates["source"] == source, "user_id"],
                candidates.loc[candidates["source"] == source, "track_id"]))
        source_recall = len(held_out_pairs & source_pairs) / len(held_out_pairs)
        print(f"{source}: {source_recall:.2%}")

    held_out_pairs = set(zip(held_out["user_id"], held_out["track_id"]))
    candidate_pairs = set(zip(candidates["user_id"], candidates["track_id"]))
    retrieved_positives = held_out_pairs & candidate_pairs
    recall = len(retrieved_positives) / len(held_out_pairs)
    print(f"\nCandidate recall: {recall:.2%}")


