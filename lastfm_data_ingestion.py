import requests
import time
import json
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://ws.audioscrobbler.com/2.0/"
API_KEY = os.getenv("API_KEY")


def call_lastfm(method, user=None, track=None, artist=None, limit=50, retries=3):
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






