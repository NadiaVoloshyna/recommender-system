from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

LISTENING_HISTORY_DIR = DATA_DIR / "listening_history"
SIMILARITIES_TRACKS_DIR = DATA_DIR / "similarities" / "tracks"
SIMILARITIES_ARTISTS_DIR = DATA_DIR / "similarities" / "artists"

