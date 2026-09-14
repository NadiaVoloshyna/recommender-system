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

# Features artifacts
FEATURES_DIR = ARTIFACTS_DIR / "features"

FULL_TRAIN_FEATURES = FEATURES_DIR / "full_train_features_50_100_50_2600.parquet"
SAMPLED_TRAIN_FEATURES = FEATURES_DIR / "sampled_train_features_50_100_50_2600.parquet"
VAL_FEATURES = FEATURES_DIR / "val_features_50_100_50_2600.parquet"
TEST_FEATURES = FEATURES_DIR / "test_features_50_100_50_2600.parquet"


