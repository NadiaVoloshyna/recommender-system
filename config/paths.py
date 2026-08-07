from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

LISTENING_HISTORY_DIR = DATA_DIR / "listening_history"
SIMILARITIES_TRACKS_DIR = DATA_DIR / "similarities" / "tracks"
SIMILARITIES_ARTISTS_DIR = DATA_DIR / "similarities" / "artists"

# Vector store artifacts
VECTOR_STORE_DIR = ARTIFACTS_DIR / "vector_store"

TRACK_INDEX_FAISS = VECTOR_STORE_DIR / "track_index.faiss"
TRACK_IDS_JSON = VECTOR_STORE_DIR / "track_ids.json"

