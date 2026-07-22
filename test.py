import pandas as pd
import os
import requests
import json
from unittest.mock import patch, Mock
from lastfm_data_ingestion import call_lastfm
from lastfm_data_ingestion import store_user_data
from seeds import create_seeds
from utils import make_id

BASE_URL = "https://ws.audioscrobbler.com/2.0/"
API_KEY = os.getenv("API_KEY")


# Check if call_lastfm() constructs the request correctly
@patch("lastfm_data_ingestion.requests.get")
def test_call_lastfm_builds_request(mock_get):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {}
    mock_get.return_value = response

    call_lastfm("user.getRecentTracks", user="NadiaV26")

    mock_get.assert_called_once_with(
        BASE_URL,
        params={
            "method": "user.getRecentTracks",
            "limit": 50,
            "api_key": API_KEY,
            "format": "json",
            "user": "NadiaV26"
        },
        timeout=10
    )


# Check call_lastfm() retry logic, simulating two failures followed by success
@patch("lastfm_data_ingestion.requests.get")
def test_call_lastfm_succeeds_after_retry(mock_get):
    failed = Mock()
    failed.raise_for_status.side_effect = requests.HTTPError()
    success = Mock()
    success.raise_for_status.return_value = None
    success.json.return_value = {
        "topartists": {
        "artist": [
            {"name": "Nirvana"}
        ]
    }
    }
    mock_get.side_effect = [failed, failed, success]

    result = call_lastfm("user.getTopArtists", user="NadiaV26")

    assert result == {
        "topartists": {
        "artist": [
            {"name": "Nirvana"}
        ]
    }
    }
    assert mock_get.call_count == 3


# Check if call_lastfm() returns None if all retry attempts fail
@ patch("lastfm_data_ingestion.requests.get")
def test_call_lastfm_returns_none_after_retries(mock_get):
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.HTTPError()
    mock_get.return_value = mock_response

    result = call_lastfm("user.getRecentTracks", user="NadiaV26")

    assert result is None
    assert mock_get.call_count == 3


# Test if store_user_data() saves successful API responses
@patch("lastfm_data_ingestion.call_lastfm")
def test_store_user_data_saves_json(mock_call_lastfm, tmp_path):
    mock_call_lastfm.return_value = {
        "topartists": {
        "artist": [
            {"name": "Nirvana"}
        ]
    }
    }

    store_user_data(
        user="NadiaV26",
        methods=["user.getTopArtists"],
        base_path=tmp_path
    )

    file_path = tmp_path / "NadiaV26" / "user.getTopArtists.json"

    assert file_path.exists()

    with open(file_path) as f:
        saved_data = json.load(f)

    assert saved_data == {
        "topartists": {
        "artist": [
            {"name": "Nirvana"}
        ]
    }
    }


# Check if store_user_data() skips failed API calls
@patch("lastfm_data_ingestion.call_lastfm")
def test_store_user_data_skips_failed_requests(mock_call_lastfm, tmp_path):
    mock_call_lastfm.return_value = None

    store_user_data(
        user="NadiaV26",
        methods=["user.getTopArtists"],
        base_path=tmp_path
    )

    file_path = tmp_path / "NadiaV26" / "user.getTopArtists.json"

    assert not file_path.exists()


# Check if store_user_data() continues after an unexpected error
@patch("lastfm_data_ingestion.call_lastfm")
def test_store_user_data_continues_after_error(mock_call_lastfm, tmp_path):
    mock_call_lastfm.side_effect = [
        Exception("API error"),
        {
            "topartists": {
                "artist": [
                    {"name": "Nirvana"}
                ]
            }
        }
    ]

    store_user_data(
        user="NadiaV26",
        methods=[
            "bad.method",
            "user.getTopArtists"
        ],
        base_path=tmp_path
    )

    good_file = (
        tmp_path /
        "NadiaV26" /
        "user.getTopArtists.json"
    )

    assert good_file.exists()


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


def test_make_id_is_deterministic():
    assert make_id("Radiohead", "artist") == make_id("Radiohead", "artist"), "make_id should return the same value for identical inputs"


def test_make_id_differs_for_prefixes():
    assert make_id("Radiohead", "artist") != make_id("Radiohead", "track"), "ids should differ when prefixes differ"

