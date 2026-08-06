from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
from features.utils import validate_columns


def build_track_embeddings(
        tracks_df: pd.DataFrame,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> dict[int, np.ndarray]:
    """
    Converts each track's metadata (artist_name + track_name) into a numerical vector
    that captures semantic information about the track.
    Validates required input columns, loads the Sentence Transformer model, processes each track, handles missing
    track and artist names, creates a text representation, generates and normalizes the embedding vector,
    and stores the embedding using the track_id as the key.
    :param tracks_df: a lookup table containing track metadata: track_name, artist_name, track_id (pd.DataFrame)
    :param model_name: the name of the pre-trained Sentence Transformer model used to generate embeddings (str)
    :return: a dictionary where: key = track_id value = normalized embedding vector (dict)
    """

    validate_columns(tracks_df, ["track_id", "artist_name", "track_name"], "tracks_df")

    model = SentenceTransformer(model_name)

    track_embeddings = {}

    for row in tracks_df.itertuples(index=False):

        track_id = row.track_id

        artist = "" if pd.isna(row.artist_name) else str(row.artist_name)
        track = "" if pd.isna(row.track_name) else str(row.track_name)

        text = f"artist: {artist} | track: {track}"

        embedding = model.encode(text)

        # Normalize for cosine similarity
        embedding = normalize([embedding])[0]

        track_embeddings[track_id] = embedding

    return track_embeddings

