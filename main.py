from data_ingestion import store_user_data, store_similar_data
from data_processing import get_users_data_paths, load_similarity_data, build_user_dataframes, build_similarity_dataframes
from global_ids import build_users_df, build_tracks_df, build_artists_df
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
    # Fetch and store raw user data from the API
    for user in USERS:
        store_user_data(user, METHODS_USER)

    # Discover stored user files and create the users lookup table
    users_files = get_users_data_paths()
    users_df = build_users_df(users_files)
    print(f"users_df:\n{users_df.head().to_string()}\n\n")
    user_lookup = dict(zip(
        users_df["username"],
        users_df["user_id"]
    ))

    # Load user JSON files and transform them into DataFrames
    user_dataframes = build_user_dataframes(users_files, user_lookup)
    recent_tracks_df = user_dataframes['recent_tracks']
    top_tracks_df = user_dataframes['top_tracks']
    top_artists_df = user_dataframes['top_artists']
    print(f"recent_tracks_df:\n{recent_tracks_df.head().to_string()}\n\n")
    print(f"top_tracks_df:\n{top_tracks_df.head().to_string()}\n\n")
    print(f"top_artists_df:\n{top_artists_df.head().to_string()}\n\n")

    # Create global IDs
    artists_df = build_artists_df(recent_tracks_df, top_tracks_df, top_artists_df)
    tracks_df = build_tracks_df(recent_tracks_df, top_tracks_df, artists_df)
    print(f"artists_df:\n{artists_df.head().to_string()}\n\n")
    print(f"tracks_df:\n{tracks_df.head().to_string()}\n\n")

    # Create seeds
    seed_tracks, seed_artists = create_seeds(top_tracks_df, top_artists_df, tracks_df, artists_df)

    # Pull and store similarity raw data for each unique seed track/artist
    for row in seed_tracks.itertuples(index=False):
        store_similar_data(
            item_id=row.track_id,
            artist_name=row.artist_name,
            track_name=row.track_name,
            category="track",
            method=METHOD_TRACK,
            base_path="similarities/tracks"
        )

    for row in seed_artists.itertuples(index=False):
        store_similar_data(
            item_id=row.artist_id,
            artist_name=row.artist_name,
            track_name=None,
            category="artist",
            method=METHOD_ARTIST,
            base_path="similarities/artists"
        )

    # Read JSON files and build similarity dataframes
    tracks_similarity_files = load_similarity_data("similarities/tracks")
    artists_similarity_files = load_similarity_data("similarities/artists")

    tracks_similarity_df = build_similarity_dataframes("track", tracks_similarity_files)
    artists_similarity_df = build_similarity_dataframes("artist", artists_similarity_files)
    print(f"Similar_tracks:\n{tracks_similarity_df.head().to_string()}\n\n")
    print(f"Similar_artists:\n{artists_similarity_df.head().to_string()}\n\n")

    # Map names to IDs, generate missing ids
    tracks_similarity_df, artists_similarity_df = map_similarity_ids(tracks_similarity_df, artists_similarity_df, tracks_df, artists_df)
    print(f"Similar_tracks:\n{tracks_similarity_df.head().to_string()}\n\n")
    print(f"Similar_artists:\n{artists_similarity_df.head().to_string()}\n\n")


if __name__ == "__main__":
    run_pipeline()
