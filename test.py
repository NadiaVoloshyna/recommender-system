import pandas as pd
import os
import requests
import json
from unittest.mock import patch, Mock
from data_ingestion import call_lastfm, store_user_data, store_similar_data
from data_processing import get_users_data_paths, build_user_dataframes, load_similarity_data
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
def test_build_tracks_df_full_transformation_pipeline(mocker):
    mocker.patch(
        "global_ids.make_id",
        return_value="track_123"
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


# create_seeds() follows the normal transformation pipeline, integration-style test
def test_create_seeds_full_transformation_pipeline():
    top_tracks_df = pd.DataFrame({
        "user_id": ["user_1", "user_1", "user_1", "user_2"],
        "track_name": ["Song A", "Song B", "Song A", "Song C"],
        "artist_name": ["Artist A", "Artist B", "Artist A", "Artist C"],
        "playcount": [100, 50, 100, 75],
    })

    top_artists_df = pd.DataFrame({
        "user_id": ["user_1", "user_1", "user_2"],
        "artist_name": ["Artist A", "Artist B", "Artist C"],
        "playcount": [200, 150, 100],
    })

    tracks_df = pd.DataFrame({
        "track_id": ["track_1", "track_2", "track_3"],
        "track_name": ["Song A", "Song B", "Song C"],
        "artist_id": ["artist_1", "artist_2", "artist_3"],
        "artist_name": ["Artist A", "Artist B", "Artist C"],
    })

    artists_df = pd.DataFrame({
        "artist_id": ["artist_1", "artist_2", "artist_3"],
        "artist_name": ["Artist A", "Artist B", "Artist C"],
    })

    seed_tracks, seed_artists = create_seeds(top_tracks_df, top_artists_df, tracks_df, artists_df)

    # Tracks are mapped correctly
    assert set(seed_tracks["track_id"]) == {"track_1", "track_2", "track_3"}
    # Artists are mapped correctly
    assert set(seed_artists["artist_id"]) == {"artist_1", "artist_2", "artist_3"}

    # Duplicate tracks are removed
    assert seed_tracks["track_id"].is_unique
    # Duplicate artists are removed
    assert seed_artists["artist_id"].is_unique

    # Output structure is correct
    assert list(seed_tracks.columns) == ["track_id", "track_name", "artist_id", "artist_name"]
    assert list(seed_artists.columns) == ["artist_id", "artist_name"]


# create_seeds() handles empty inputs
def test_create_seeds_empty_inputs():
    top_tracks_df = pd.DataFrame({
        "user_id": [],
        "track_name": [],
        "artist_name": [],
        "playcount": [],
    })

    top_artists_df = pd.DataFrame({
        "user_id": [],
        "artist_name": [],
        "playcount": [],
    })

    tracks_df = pd.DataFrame({
        "track_id": [],
        "track_name": [],
        "artist_id": [],
        "artist_name": [],
    })

    artists_df = pd.DataFrame({
        "artist_id": [],
        "artist_name": [],
    })

    seed_tracks, seed_artists = create_seeds(top_tracks_df, top_artists_df, tracks_df, artists_df)

    assert seed_tracks.empty
    assert seed_artists.empty


# create_seeds() removes tracks and artists that fail lookup joins
def test_create_seeds_removes_unknown_tracks_and_artists():
    top_tracks_df = pd.DataFrame({
        "user_id": ["user_1"],
        "track_name": ["Unknown Song"],
        "artist_name": ["Unknown Artist"],
        "playcount": [100],
    })

    top_artists_df = pd.DataFrame({
        "user_id": ["user_1"],
        "artist_name": ["Unknown Artist"],
        "playcount": [100],
    })

    tracks_df = pd.DataFrame({
        "track_id": ["track_1"],
        "track_name": ["Known Song"],
        "artist_id": ["artist_1"],
        "artist_name": ["Known Artist"],
    })

    artists_df = pd.DataFrame({
        "artist_id": ["artist_1"],
        "artist_name": ["Known Artist"],
    })

    seed_tracks, seed_artists = create_seeds(top_tracks_df, top_artists_df, tracks_df, artists_df)

    assert seed_tracks.empty
    assert seed_artists.empty


# store_similar_data() saves track similarity responses correctly
def test_store_similar_data_saves_track_similarity(mocker, tmp_path):
    mocker.patch(
        "data_ingestion.call_lastfm",
        return_value={
            "similartracks": {
                "track": [
                    {"name": "Song B"}
                ]
            }
        },
    )

    store_similar_data(
        item_id="track_123",
        artist_name="Coldplay",
        category="track",
        method="track.getSimilar",
        base_path=tmp_path,
        track_name="Yellow",
    )

    file_path = tmp_path / "track_123.json"
    assert file_path.exists()

    with open(file_path, encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data == {
        "similartracks": {
            "track": [
                {"name": "Song B"}
            ]
        }
    }


# store_similar_data() saves artist similarity responses correctly
def test_store_similar_data_saves_artist_similarity(mocker, tmp_path):
    mocker.patch(
        "data_ingestion.call_lastfm",
        return_value={
            "similarartists": {
                "artist": [
                    {"name": "Muse"}
                ]
            }
        },
    )

    store_similar_data(
        item_id="artist_123",
        artist_name="Coldplay",
        category="artist",
        method="artist.getSimilar",
        base_path=tmp_path,
    )

    file_path = tmp_path / "artist_123.json"
    assert file_path.exists()

    with open(file_path, encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data["similarartists"]["artist"][0]["name"] == "Muse"


# store_similar_data() handles empty API responses
def test_store_similar_data_handles_no_response(mocker, tmp_path):
    mocker.patch(
        "data_ingestion.call_lastfm",
        return_value=None,
    )

    store_similar_data(
        item_id="track_123",
        artist_name="Coldplay",
        category="track",
        method="track.getSimilar",
        base_path=tmp_path,
        track_name="Yellow",
    )

    assert list(tmp_path.iterdir()) == []


# store_similar_data() handles API errors
def test_store_similar_data_handles_api_error(mocker, tmp_path):
    mocker.patch(
        "data_ingestion.call_lastfm",
        return_value={
            "error": 6,
            "message": "Track not found",
        },
    )

    store_similar_data(
        item_id="track_123",
        artist_name="Unknown",
        category="track",
        method="track.getSimilar",
        base_path=tmp_path,
        track_name="Unknown Song",
    )

    assert list(tmp_path.iterdir()) == []


# store_similar_data() handles unexpected exceptions
def test_store_similar_data_handles_exception(mocker, tmp_path):
    mocker.patch(
        "data_ingestion.call_lastfm",
        side_effect=Exception("API failure"),
    )

    # Should not raise an exception
    store_similar_data(
        item_id="track_123",
        artist_name="Coldplay",
        category="track",
        method="track.getSimilar",
        base_path=tmp_path,
        track_name="Yellow",
    )

    assert list(tmp_path.iterdir()) == []


# load_similarity_data() returns file paths
def test_load_similarity_data_returns_files(tmp_path):
    (tmp_path / "track_123.json").write_text("{}")
    (tmp_path / "track_456.json").write_text("{}")

    result = load_similarity_data(str(tmp_path))

    assert len(result) == 2

    assert str(tmp_path / "track_123.json") in result
    assert str(tmp_path / "track_456.json") in result


# load_similarity_data() handles missing directory
def test_load_similarity_data_missing_directory(tmp_path):
    missing_path = tmp_path / "does_not_exist"

    result = load_similarity_data(str(missing_path))

    assert result == []


# load_similarity_data() handles empty directory
def test_load_similarity_data_empty_directory(tmp_path):
    result = load_similarity_data(str(tmp_path))

    assert result == []


def test_make_id_is_deterministic():
    assert make_id("Radiohead", "artist") == make_id("Radiohead", "artist"), "make_id should return the same value for identical inputs"


def test_make_id_differs_for_prefixes():
    assert make_id("Radiohead", "artist") != make_id("Radiohead", "track"), "ids should differ when prefixes differ"

