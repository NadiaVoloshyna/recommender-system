import hashlib


def make_id(x: str, prefix: str) -> str:
    key = f"{prefix}:{x}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
