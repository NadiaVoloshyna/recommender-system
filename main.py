from data_ingestion import store_user_data, store_similar_data
from data_processing import \
    get_users_data_paths, \
    load_similarity_data, \
    build_user_dataframes, \
    build_track_similarity_dataframe, \
    build_artist_similarity_dataframe
from global_ids import build_users_df, build_tracks_df, build_artists_df
from seeds import create_seeds
from tabulate import tabulate

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


def run_pipeline(fetch_api_data=False):
    print("\nStarting Music Recommender ETL Pipeline\n")

    try:
        if fetch_api_data:
            print("1 Fetching user listening data...")
            # Fetch raw user data from the API and persist it locally
            for user in USERS:
                store_user_data(user, METHODS_USER)
            print("User data stored\n")
        else:
            print("1 Using existing user listening data...\n")

        print("2 Building user lookup table...")
        # Locate stored user data files and build the user lookup table
        user_files = get_users_data_paths()
        _, user_lookup = build_users_df(user_files)
        print(f"Users processed: {len(user_lookup)}\n")

        print("3 Building user interaction DataFrames...")
        # Load user JSON data and convert each dataset into DataFrames
        user_dfs = build_user_dataframes(user_files, user_lookup)
        recent_tracks_df = user_dfs['recent_tracks']
        top_tracks_df = user_dfs['top_tracks']
        top_artists_df = user_dfs['top_artists']
        print("DataFrames created:")

        for name, df in {
            "Recent Tracks": recent_tracks_df,
            "Top Tracks": top_tracks_df,
            "Top Artists": top_artists_df,
        }.items():
            print(f"\n{name}\n")
            print(tabulate(df.head(5), headers="keys", tablefmt="fancy_grid", showindex=False))

        print("\n4 Creating artist and track mappings...")
        # Build global artist and track IDs with lookup mappings
        artists_df, artist_lookup = build_artists_df(recent_tracks_df, top_tracks_df, top_artists_df)
        tracks_df, track_lookup = build_tracks_df(recent_tracks_df, top_tracks_df, artists_df)
        print(f"Tracks: {len(tracks_df):,}")
        print(f"Artists: {len(artists_df):,}\n")

        print("5 Generating similarity seeds...")
        # Generate seed tracks and artists for similarity searches
        seed_tracks, seed_artists = create_seeds(top_tracks_df, top_artists_df, tracks_df, artists_df)
        print(f"Seed tracks: {len(seed_tracks):,}")
        print(f"Seed artists: {len(seed_artists):,}\n")

        if fetch_api_data:
            print("6 Fetching similarity data...")
            # Fetch and store raw similarity data for each unique seed track
            for row in seed_tracks.itertuples(index=False):
                store_similar_data(
                    item_id=row.track_id,
                    artist_name=row.artist_name,
                    track_name=row.track_name,
                    category="track",
                    method=METHOD_TRACK,
                    base_path="similarities/tracks"
                )
            # Fetch and store raw similarity data for each unique seed artist
            for row in seed_artists.itertuples(index=False):
                store_similar_data(
                    item_id=row.artist_id,
                    artist_name=row.artist_name,
                    track_name=None,
                    category="artist",
                    method=METHOD_ARTIST,
                    base_path="similarities/artists"
                )
            print("Similarity data collected\n")
        else:
            print("6 Using existing similarity data...\n")

        print("7 Building similarity datasets...")
        # Load stored similarity JSON files and convert them into DataFrames with global track and artist IDs
        track_similarity_files = load_similarity_data("similarities/tracks")
        artist_similarity_files = load_similarity_data("similarities/artists")

        track_similarity_df = build_track_similarity_dataframe(track_similarity_files, track_lookup)
        artist_similarity_df = build_artist_similarity_dataframe(artist_similarity_files, artist_lookup)

        for name, df in {
            "Track Similarity DataFrame": track_similarity_df,
            "Artist Similarity DataFrame": artist_similarity_df,
        }.items():
            print(f"\n{name}\n")
            print(tabulate(df.head(5), headers="keys", tablefmt="fancy_grid", showindex=False))

        print("\n Pipeline completed successfully!")

    except Exception as e:
        print("\n Pipeline failed")
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    run_pipeline()
