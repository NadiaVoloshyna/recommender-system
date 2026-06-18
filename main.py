from lastfm_data_ingestion import store_user_data, store_similar_data
from data_preprocessing import load_user_data, load_similarity_data, build_user_dataframes, build_similarity_dataframes
from ids import build_tracks_df, build_artists_df
from seeds import create_seeds
from map_similarity_ids import map_similarity_ids

USERS = ["NadiaV26", "SkullyXIX", "FabioBrt", "owenisupercool", "loomingcloset",
         "Btree15", "Burzay8", "arham23213", "Jeffrylol", "kkauabr",
         "yashfrr", "dangernz123", "Realejt", "Prawin1107", "Majortar",
         "julianmrios", "beanzzboi", "elden98", "mayo-143", "Muellera3",
         "Truesify", "Pklore", "erqnhrt", "Noora04", "enizekkj",
         "joaco_alv", "Vast_Venos", "Pudimkjjk", "FRIESDAY24", "goodboyryley",
         "LedMosleu", "t1nt3dc14w", "ShikuYoki", "Bannjo", "AndrewSilvs",
         "Peepee99", "potato_nuggetz", "A-rat", "MatVilar", "Stella141",
         "mreowsaurusrex", "malta777", "NerFixMusic", "PickleRick9054", "JOKAAU",
         "berkinsecme", "Isaakalphaigk", "Bury210", "Jona161", "LucienCalu",
         "RightEazy"]
METHODS_USER = [
    "user.getRecentTracks",
    "user.getTopTracks",
    "user.getTopArtists"
]
METHOD_TRACK = "track.getSimilar"
METHOD_ARTIST = "artist.getSimilar"


def run_pipeline():
    # Pull and store user raw data
    for user in USERS:
        store_user_data(user, METHODS_USER)

    # Read user JSON files and build dataframes
    user_files = load_user_data()
    user_dataframes = build_user_dataframes(user_files)
    recent_tracks_df = user_dataframes['recent_tracks']   # track_name, artist_name, timestamp, url, user, user_id
    top_tracks_df = user_dataframes['top_tracks']         # track_name, artist_name, playcount, url, user, user_id
    top_artists_df = user_dataframes['top_artists']       # artist_name, playcount, url, user, user_id
    print(f"recent_tracks_df:\n{recent_tracks_df.head().to_string()}\n\n")
    print(f"top_tracks_df:\n{top_tracks_df.head().to_string()}\n\n")
    print(f"top_artists_df:\n{top_artists_df.head().to_string()}\n\n")

    # Create global IDs
    artists_df = build_artists_df(recent_tracks_df, top_tracks_df, top_artists_df)   # artist_name, artist_id
    tracks_df = build_tracks_df(recent_tracks_df, top_tracks_df, artists_df)         # track_name, artist_name, track_id, artist_id
    print(f"artists_df:\n{artists_df.head().to_string()}\n\n")
    print(f"tracks_df:\n{tracks_df.head().to_string()}\n\n")

    # Create seeds
    seed_tracks, seed_artists = create_seeds(top_tracks_df, top_artists_df, tracks_df, artists_df) # track_id, artist_id | list of ids

    # Pull and store similarity raw data for each unique seed track/artist
    for row in seed_tracks.itertuples(index=False):
        track_id = row.track_id
        artist_id = row.artist_id

        track_name = tracks_df.loc[tracks_df["track_id"] == track_id, "track_name"].iloc[0]
        artist_name = artists_df.loc[artists_df["artist_id"] == artist_id, "artist_name"].iloc[0]

        store_similar_data(
            item_id=track_id,
            artist_name=artist_name,
            category="track",
            method=METHOD_TRACK,
            base_path="similarities/tracks",
            track_name=track_name
        )

    for artist_id in seed_artists:
        artist_name = artists_df.loc[artists_df["artist_id"] == artist_id, "artist_name"].iloc[0]
        store_similar_data(
            item_id=artist_id,
            artist_name=artist_name,
            category="artist",
            method=METHOD_ARTIST,
            base_path="similarities/artists",
            track_name=None
        )

    # Read JSON files and build similarity dataframes
    tracks_similarity_files = load_similarity_data("similarities/tracks")
    artists_similarity_files = load_similarity_data("similarities/artists")

    tracks_similarity_df = build_similarity_dataframes("track", tracks_similarity_files)      # track_id, similar_track_name, similarity_score
    artists_similarity_df = build_similarity_dataframes("artist", artists_similarity_files)   # artist_id, similar_artist_name, similarity_score
    print(f"Similar_tracks:\n{tracks_similarity_df.head().to_string()}\n\n")
    print(f"Similar_artists:\n{artists_similarity_df.head().to_string()}\n\n")

    # Map names to IDs, generate missing ids
    tracks_similarity_df, artists_similarity_df = map_similarity_ids(tracks_similarity_df, artists_similarity_df, tracks_df, artists_df)
    print(f"Similar_tracks:\n{tracks_similarity_df.head().to_string()}\n\n")
    print(f"Similar_artists:\n{artists_similarity_df.head().to_string()}\n\n")


if __name__ == "__main__":
    run_pipeline()
