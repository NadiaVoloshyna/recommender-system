from vector_store.faiss_builder import build_faiss


def run_vector_store_pipeline(track_embeddings):
    try:
        faiss_index, track_id_mapping = build_faiss(track_embeddings)

        print("\nFAISS Index\n")
        print(f"Number of vectors: {faiss_index.ntotal}")
        print(f"Embedding dimension: {faiss_index.d}")
        print(f"Number of track IDs: {len(track_id_mapping)}")

        print("\nFAISS ID mapping sample\n")
        for i, track_id in enumerate(track_id_mapping[:10]):
            print(i, track_id)

        return faiss_index, track_id_mapping

    except Exception as e:
        print("\n Pipeline failed")
        print(f"Error: {e}")
        raise




