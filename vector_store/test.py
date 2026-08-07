import json
import numpy as np
import faiss
import pytest

from vector_store.faiss_builder import build_faiss
from config.paths import TRACK_INDEX_FAISS, TRACK_IDS_JSON


# Creates normalized test embeddings
@pytest.fixture
def sample_embeddings():
    embeddings = {
        "track_001": np.array([1.0, 0.0, 0.0]),
        "track_002": np.array([0.0, 1.0, 0.0]),
        "track_003": np.array([0.0, 0.0, 1.0]),
    }

    return embeddings


# build_faiss() creates FAISS index that contains vectors
def test_build_faiss_creates_index(sample_embeddings):
    index, track_ids = build_faiss(sample_embeddings)

    assert isinstance(index, faiss.Index)
    assert index.ntotal == 3
    assert track_ids == [
        "track_001",
        "track_002",
        "track_003"
    ]


# build_faiss(): FAISS positions match track_ids order
def test_build_faiss_preserves_track_order(sample_embeddings):
    index, track_ids = build_faiss(sample_embeddings)

    query = np.array(
        [[1.0, 0.0, 0.0]],
        dtype="float32"
    )

    scores, indices = index.search(query, 1)

    nearest_position = indices[0][0]

    assert track_ids[nearest_position] == "track_001"


# build_faiss(): empty dictionary fails
def test_build_faiss_empty_embeddings():
    with pytest.raises(ValueError, match="track_embeddings is empty"):
        build_faiss({})


# build_faiss() saves FAISS index and ID mapping
def test_build_faiss_saves_index(sample_embeddings):
    build_faiss(sample_embeddings)

    assert TRACK_INDEX_FAISS.exists()
    assert TRACK_IDS_JSON.exists()


# build_faiss(): saved FAISS index is readable
def test_saved_faiss_can_be_loaded(sample_embeddings):
    build_faiss(sample_embeddings)

    loaded_index = faiss.read_index(str(TRACK_INDEX_FAISS))

    with open(TRACK_IDS_JSON) as f:
        loaded_ids = json.load(f)

    assert loaded_index.ntotal == 3
    assert loaded_ids == [
        "track_001",
        "track_002",
        "track_003"
    ]