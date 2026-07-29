import pandas as pd
import os
import requests
import json
from unittest.mock import patch, Mock
from data_ingestion import call_lastfm, store_user_data
from data_processing import get_users_data_paths, build_user_dataframes
from global_ids import build_users_df, build_artists_df, build_tracks_df
from seeds import create_seeds
from utils import make_id

BASE_URL = "https://ws.audioscrobbler.com/2.0/"
API_KEY = os.getenv("API_KEY")


# call_lastfm() constructs the request correctly
@patch("data_ingestion.requests.get")
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
@patch("data_ingestion.requests.get")
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


# call_lastfm() returns None if all retry attempts fail
@ patch("data_ingestion.requests.get")
def test_call_lastfm_returns_none_after_retries(mock_get):
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.HTTPError()
    mock_get.return_value = mock_response

    result = call_lastfm("user.getRecentTracks", user="NadiaV26")

    assert result is None
    assert mock_get.call_count == 3


# store_user_data() saves successful API responses
@patch("data_ingestion.call_lastfm")
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


# store_user_data() skips failed API calls
@patch("data_ingestion.call_lastfm")
def test_store_user_data_skips_failed_requests(mock_call_lastfm, tmp_path):
    mock_call_lastfm.return_value = None

    store_user_data(
        user="NadiaV26",
        methods=["user.getTopArtists"],
        base_path=tmp_path
    )

    file_path = tmp_path / "NadiaV26" / "user.getTopArtists.json"

    assert not file_path.exists()


# store_user_data() continues after an unexpected error
@patch("data_ingestion.call_lastfm")
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


# get_user_data_paths() finds files inside user folders
def test_get_user_data_paths_returns_files(tmp_path):
    user_folder = tmp_path / "NadiaV26"
    user_folder.mkdir()

    file1 = user_folder / "user.getRecentTracks.json"
    file2 = user_folder / "user.getTopArtists.json"

    file1.write_text("{}")
    file2.write_text("{}")

    result = get_users_data_paths(tmp_path)

    assert file1.as_posix() in [path.replace("\\", "/") for path in result]
    assert file2.as_posix() in [path.replace("\\", "/") for path in result]


# get_user_data_paths() ignores files directly inside base folder
def test_get_user_data_paths_ignores_files_in_base_folder(tmp_path):
    user_folder = tmp_path / "NadiaV26"
    user_folder.mkdir()

    user_file = user_folder / "user.getRecentTracks.json"
    user_file.write_text("{}")

    ignored_file = tmp_path / "README.txt"
    ignored_file.write_text("ignore me")

    result = get_users_data_paths(tmp_path)

    assert str(user_file) in result
    assert str(ignored_file) not in result


# get_user_data_paths() returns empty list when no user data exists
def test_get_user_data_paths_returns_empty_list(tmp_path):
    result = get_users_data_paths(tmp_path)

    assert result == []


# Check that valid file paths produce the expected DataFrame in build_users_df()
def test_build_users_df_creates_dataframe():
    users_files = [
        "/raw_data/alice/file1.json",
        "/raw_data/bob/file2.json"
    ]

    result = build_users_df(users_files)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["user_id", "username"]
    assert len(result) == 2


# build_users_df() removes users duplicates users
def test_build_users_df_removes_duplicate_users():
    users_files = [
        "/raw_data/alice/file1.json",
        "/raw_data/alice/file2.json",
        "/raw_data/bob/file1.json"
    ]

    result = build_users_df(users_files)

    assert len(result) == 2
    assert set(result["username"]) == {"alice", "bob"}


# Usernames are sorted in build_users_df()
def test_build_users_df_sorts_users():
    users_files = [
        "/raw_data/charlie/file.json",
        "/raw_data/alice/file.json",
        "/raw_data/bob/file.json"
    ]

    result = build_users_df(users_files)

    assert list(result["username"]) == [
        "alice",
        "bob",
        "charlie"
    ]


# User IDs are generated in build_users_df()
def test_build_users_df_generates_ids(mocker):
    mocker.patch(
        "global_ids.make_id",
        return_value="user_123"
    )

    result = build_users_df(["/raw_data/alice/file.json"])

    assert result.iloc[0]["user_id"] == "user_123"


# An empty list should return an empty DataFrame in build_users_df()
def test_build_users_df_empty_input():
    result = build_users_df([])

    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == []


# Check build_user_dataframes() main extraction logic, recent tracks extraction
def test_recent_tracks_parsing(tmp_path):
    user_folder = tmp_path / "NadiaV26"
    user_folder.mkdir()
    user_file = user_folder / "user.getRecentTracks.json"

    data = {
        "recenttracks": {
            "track": [
                {
                    "artist": {"#text": "AC/DC"},
                    "name": "Thunderstruck",
                    "date": {"uts": "123456"}
                }
            ]
        }
    }

    user_file.write_text(json.dumps(data))

    result = build_user_dataframes(
        [str(user_file)],
        {"NadiaV26": 1}
    )

    df = result["recent_tracks"]

    assert len(df) == 1
    assert df.iloc[0]["user_id"] == 1
    assert df.iloc[0]["user"] == "NadiaV26"
    assert df.iloc[0]["track_name"] == "Thunderstruck"
    assert df.iloc[0]["artist_name"] == "AC/DC"
    assert df.iloc[0]["timestamp"] == "123456"


# Check build_user_dataframes() main extraction logic, top tracks extraction, playcount conversion
def test_top_tracks_parsing(tmp_path):
    user_folder = tmp_path / "NadiaV26"
    user_folder.mkdir()
    user_file = user_folder / "user.getTopTracks.json"

    data = {
        "toptracks": {
            "track": [
                {
                    "name": "Back in Black",
                    "artist": {"name": "AC/DC"},
                    "playcount": "50"
                }
            ]
        }
    }

    user_file.write_text(json.dumps(data))

    result = build_user_dataframes(
        [str(user_file)],
        {"NadiaV26": 1}
    )

    df = result["top_tracks"]

    assert len(df) == 1
    assert df.iloc[0]["track_name"] == "Back in Black"
    assert df.iloc[0]["artist_name"] == "AC/DC"
    assert df.iloc[0]["playcount"] == 50
    assert pd.api.types.is_integer_dtype(df["playcount"])


# Check build_user_dataframes() main extraction logic, top artists parsing, rank/playcount extraction
def test_top_artists_parsing(tmp_path):
    user_folder = tmp_path / "NadiaV26"
    user_folder.mkdir()
    user_file = user_folder / "user.getTopArtists.json"

    data = {
        "topartists": {
            "artist": [
                {
                    "name": "AC/DC",
                    "playcount": "100",
                    "@attr": {"rank": "1"}
                }
            ]
        }
    }

    user_file.write_text(json.dumps(data))

    result = build_user_dataframes(
        [str(user_file)],
        {"NadiaV26": 1}
    )

    df = result["top_artists"]

    assert len(df) == 1
    assert df.iloc[0]["artist_name"] == "AC/DC"
    assert df.iloc[0]["playcount"] == 100
    assert df.iloc[0]["rank"] == 1


# build_user_dataframes() removes invalid rows, data cleaning
def test_invalid_rows_are_removed(tmp_path):
    user_folder = tmp_path / "NadiaV26"
    user_folder.mkdir()
    user_file = user_folder / "user.getRecentTracks.json"

    data = {
        "recenttracks": {
            "track": [
                {
                    "artist": {"#text": ""},
                    "name": "",
                    "date": {"uts": "123456"}
                },
                {
                    "artist": {"#text": "Valid Artist"},
                    "name": "Valid Song",
                    "date": {"uts": "123456"}
                }
            ]
        }
    }

    user_file.write_text(json.dumps(data))

    result = build_user_dataframes(
        [str(user_file)],
        {"NadiaV26": 1}
    )

    df = result["recent_tracks"]

    assert len(df) == 1
    assert df.iloc[0]["track_name"] == "Valid Song"


# build_user_dataframes() handles empty inputs
def test_empty_api_response_returns_empty_dataframe(tmp_path):
    user_folder = tmp_path / "NadiaV26"
    user_folder.mkdir()
    user_file = user_folder / "user.getRecentTracks.json"

    data = {
        "recenttracks": {
            "track": []
        }
    }

    user_file.write_text(json.dumps(data))

    result = build_user_dataframes(
        [str(user_file)],
        {"NadiaV26": 1}
    )

    assert result["recent_tracks"].empty


# build_user_dataframes() combines multiple users
def test_multiple_users_are_combined(tmp_path):
    files = []

    for username in ["alice", "bob"]:
        user_folder = tmp_path / username
        user_folder.mkdir()
        user_file = user_folder / "user.getTopTracks.json"

        data = {
            "toptracks": {
                "track": [
                    {
                        "name": f"{username} song",
                        "artist": {"name": "Artist"},
                        "playcount": "10"
                    }
                ]
            }
        }

        user_file.write_text(json.dumps(data))
        files.append(str(user_file))

    result = build_user_dataframes(
        files,
        {
            "alice": 1,
            "bob": 2
        }
    )

    df = result["top_tracks"]

    assert len(df) == 2
    assert set(df["user"]) == {"alice", "bob"}


# build_artists_df follows the normal transformation pipeline, integration-style test
def test_build_artists_df_full_transformation_pipeline(mocker):
    mocker.patch(
        "global_ids.make_id",
        return_value="artist_123"
    )
    recent_tracks_df = pd.DataFrame({"artist_name": ["Coldplay", "Muse", "Coldplay", None]})
    top_tracks_df = pd.DataFrame({"artist_name": ["Muse", "Radiohead", None]})
    top_artists_df = pd.DataFrame({"artist_name": ["Coldplay", "Oasis"]})

    result = build_artists_df(recent_tracks_df, top_tracks_df, top_artists_df)

    # All sources are combined
    assert set(result["artist_name"]) == {
        "Coldplay",
        "Muse",
        "Radiohead",
        "Oasis",
    }

    # Duplicates are removed
    assert len(result) == 4
    assert result["artist_name"].is_unique

    # Missing values are removed
    assert result["artist_name"].isna().sum() == 0

    # IDs are generated
    assert (result["artist_id"] == "artist_123").all()

    # Verify lookup table structure
    assert list(result.columns) == ["artist_id", "artist_name"]


# build_artists_df handles empty inputs
def test_build_artists_df_empty_inputs():
    recent_tracks_df = pd.DataFrame({"artist_name": []})
    top_tracks_df = pd.DataFrame({"artist_name": []})
    top_artists_df = pd.DataFrame({"artist_name": []})

    result = build_artists_df(recent_tracks_df, top_tracks_df, top_artists_df)

    assert result.empty


# build_tracks_df follows the normal transformation pipeline, integration-style test
def test_build_artists_df_full_transformation_pipeline(mocker):
    mocker.patch(
        "global_ids.make_id",
        return_value="artist_123"
    )

    recent_tracks_df = pd.DataFrame({
        "track_name": ["Yellow", "Uprising", "Yellow", None],
        "artist_name": ["Coldplay", "Muse", "Coldplay", "Muse"],
    })
    top_tracks_df = pd.DataFrame({
        "track_name": ["Yellow", "Paranoid Android"],
        "artist_name": ["Coldplay", "Radiohead"],
    })
    artists_df = pd.DataFrame({
        "artist_name": ["Coldplay", "Muse"],
        "artist_id": ["artist_1", "artist_2"],
    })

    result = build_tracks_df(recent_tracks_df, top_tracks_df, artists_df)

    # Missing values are removed
    assert result["track_name"].isna().sum() == 0
    assert result["artist_name"].isna().sum() == 0

    # Duplicates are removed
    assert len(result) == 2
    assert result.duplicated(subset=["track_name", "artist_name"]).sum() == 0

    # Artist IDs are attached
    assert set(result["artist_id"]) == {"artist_1", "artist_2"}

    # Tracks from known artists are retained
    assert set(result["track_name"]) == {"Yellow", "Uprising"}
    assert "Radiohead" not in result["artist_name"].values

    # Track IDs are generated
    assert (result["track_id"] == "track_123").all()

    # Verify lookup table structure
    assert list(result.columns) == ["track_id", "track_name", "artist_id", "artist_name"]


# build_tracks_df handles empty inputs
def test_build_tracks_df_empty_inputs():
    recent_tracks_df = pd.DataFrame({"track_name": [], "artist_name": []})
    top_tracks_df = pd.DataFrame({"track_name": [], "artist_name": []})
    artists_df = pd.DataFrame({"artist_name": [], "artist_id": []})

    result = build_tracks_df(recent_tracks_df, top_tracks_df, artists_df)

    assert result.empty


"""
--------------------------------------------------------------------------------------------------------------------
"""


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

