import pandas as pd
from main import run_pipeline
from unittest.mock import patch
from utils import make_id
from seeds import create_seeds


def test_make_id_is_deterministic():
    assert make_id("Radiohead", "artist") == make_id("Radiohead", "artist"), "make_id should return the same value for identical inputs"


def test_make_id_differs_for_prefixes():
    assert make_id("Radiohead", "artist") != make_id("Radiohead", "track"), "ids should differ when prefixes differ"


def test_create_seeds():
    top_tracks_df = pd.DataFrame({  # track_name, artist_name, playcount, url, user, user_id
        "track_name": ["Lovefool", "Invisible", "the man who sold the world - live", "Snake Eater", "Everlong"],
        "artist_name": ["The Cardigans", "Duran Duran", "Nirvana", "Cynthia Harrell", "Foo Fighters"],
        "playcount": [116, 12, 11, 9, 9],
        "user_id": ["42b06c7e091d4ca58e14929a4f8b315b16c9559c29d0615b7b22e192bad8dd54"] * 5
    })
    top_artists_df = pd.DataFrame({  # artist_name, playcount, url, user, user_id
        "artist_name": ["The Cardigans", "The Weeknd", "Nirvana", "Kevin Sherwood", "Fall Out Boy"],
        "playcount": [116, 108, 84, 42, 25],
        "user_id": ["42b06c7e091d4ca58e14929a4f8b315b16c9559c29d0615b7b22e192bad8dd54"] * 5
    })
    tracks_df = pd.DataFrame({  # track_name, artist_name, track_id, artist_id
        "track_name": ["Heart-Shaped Box", "Somebody Told Me", "Sweet Child o' Mine", "Fortunate Son", "Snake Eater"],
        "artist_name": ["Nirvana", "The Killers", "Guns N' Roses", "Creedence Clearwater Revival", "Cynthia Harrell"],
        "track_id": ["a", "ab", "abc", "abcd", "abcde"],
        "artist_id": ["1", "12", "123", "1234", "12345"]
    })
    artists_df = pd.DataFrame({        # artist_name, artist_id
        "artist_name": ["The Cardigans", "The Weeknd", "Nirvana", "Kevin Sherwood", "Fall Out Boy"],
        "artist_id": [
            "3fb3d8fa99488c491008d555baa494ad3722a939e1ec66e66b9010ce7db5d1ea",
            "82cd882e512362280247750e52ff46b1e8be2064f0a2668f5838573e2c042dea",
            "5384b648be9b9e12333c4d2ebc49ab51ad049a15fa6623ea1884e9a25014f219",
            "1b0904879cf52adc6192f8db596dedcc85a98d022baed4561130faf682571132",
            "3c6c6b248084a76e34d03d5040c9e069a161033203dec1977d17c7f9d24f31a8"
        ]
    })

    unique_track_ids, unique_artist_ids = create_seeds(top_tracks_df, top_artists_df, tracks_df, artists_df)

    # Unmatched tracks are removed after the merge.
    assert len(unique_track_ids) == 1

    row = unique_track_ids.iloc[0]
    assert row["track_id"] == "abcde"
    assert row["artist_id"] == "12345"

    # All artists are matched.
    assert len(unique_artist_ids) == 5

    assert set(unique_artist_ids) == {
        "3fb3d8fa99488c491008d555baa494ad3722a939e1ec66e66b9010ce7db5d1ea",
        "82cd882e512362280247750e52ff46b1e8be2064f0a2668f5838573e2c042dea",
        "5384b648be9b9e12333c4d2ebc49ab51ad049a15fa6623ea1884e9a25014f219",
        "1b0904879cf52adc6192f8db596dedcc85a98d022baed4561130faf682571132",
        "3c6c6b248084a76e34d03d5040c9e069a161033203dec1977d17c7f9d24f31a8",
    }

# @patch("main.store_user_data")
# def test_pipeline_calls_store_user_data(mock_store):
#     run_pipeline()
#     mock_store.assert_called_once()


def main():
    test_make_id_is_deterministic()
    test_make_id_differs_for_prefixes()
    test_create_seeds()


if __name__ == "__main__":
    main()


