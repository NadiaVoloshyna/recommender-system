import json
import faiss
import numpy as np
from config.paths import TRACK_INDEX_FAISS, TRACK_IDS_JSON


def build_faiss(track_embeddings: dict[str, np.ndarray]) -> tuple[faiss.Index, list[str]]:
    """
    Creates a FAISS index from track embeddings.
    :param track_embeddings: dictionary where keys are track IDs and values are normalized embedding vectors (dict)
    :return: tuple containing:
        index: FAISS index containing all track embeddings (faiss.Index)
        track_ids: track IDs in the same order as the vectors stored in the index (list)
    """
    if not track_embeddings:
        raise ValueError("track_embeddings is empty")

    # Retrieve track IDs
    track_ids = list(track_embeddings.keys())

    # Retrieve embedding vectors
    vectors = np.vstack(list(track_embeddings.values())).astype("float32")

    # Create FAISS index(Inner Product = Cosine Similarity for normalized vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])

    # Add all vectors to the index
    index.add(vectors)

    # Ensure artifact directory exists
    TRACK_INDEX_FAISS.parent.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, str(TRACK_INDEX_FAISS))

    # Save mapping between FAISS IDs and track IDs
    with open(TRACK_IDS_JSON, "w") as f:
        json.dump(track_ids, f)

    return index, track_ids


