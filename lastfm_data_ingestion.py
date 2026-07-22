import requests
import time
import json
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://ws.audioscrobbler.com/2.0/"
API_KEY = os.getenv("API_KEY")


def call_lastfm(
        method: str,
        user: str = None,
        track: str = None,
        artist: str = None,
        limit: int = 50,
        retries: int = 3
) -> dict | None:
    """
    Creates the request parameters, adds optional parameters, sends the API request, checks for errors,
    retries if the request fails.
    :param method: Last.fm API method to call (e.g., "user.getTopTracks")(str)
    :param user: Last.fm username for user-specific requests (str, optional)
    :param track: track name for track-related requests (str, optional)
    :param artist: artist name for artist-related requests (str, optional)
    :param limit: maximum number of results to return, defaults 50 (int, optional)
    :param retries: number of retry attempts if the request fails, defaults 3 (int, optional)
    :return: parsed JSON response if successful (dict), none if all retry attempts fail
    """
    params = {
        "method": method,
        "limit": limit,
        "api_key": API_KEY,
        "format": "json"
    }
    if user:
        params["user"] = user
    if track:
        params["track"] = track
    if artist:
        params["artist"] = artist

    for attempt in range(retries):
        try:
            response = requests.get(BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed for {method}:", e)

            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


# User data ingestion
def store_user_data(user, methods, base_path="raw_data"):
    # Create user directory
    user_path = os.path.join(base_path, user)
    os.makedirs(user_path, exist_ok=True)

    for method in methods:
        try:
            data = call_lastfm(method, user=user)
            if data is None:
                continue

            filename = f"{method}.json"
            file_path = os.path.join(user_path, filename)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Error fetching {method} for {user}: {e}")

    return


# Similarity data ingestion
def store_similar_data(item_id, artist_name, category, method, base_path, track_name=None):
    try:
        if category == "track":
            data = call_lastfm(method, track=track_name, artist=artist_name)
        else:
            data = call_lastfm(method, artist=artist_name)

        if data is None:
            return
        if "error" in data:
            print(f"message={data['message']}")
            return

        filename = f"{item_id}.json"
        file_path = os.path.join(base_path, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        print(f"Error fetching {method} for {item_id} in {category}: {e}")

    return






