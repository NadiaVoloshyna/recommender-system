import hashlib
import pandas as pd
import unicodedata


def make_id(x: str, prefix: str) -> str:
    key = f"{prefix}:{x}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def clean_categorical(series):
    if not isinstance(series, pd.Series):
        return pd.Series(dtype="string")

    s = series.astype("string")
    s = s.map(
        lambda x: (
            unicodedata.normalize("NFKD", x)
            .encode("ascii", "ignore")
            .decode("utf-8")
            .strip()
            .lower()
            if pd.notna(x)
            else x
        )
    )

    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.mask(s == "", pd.NA)

    return s


def clean_numerical(series):
    if not isinstance(series, pd.Series):
        return pd.Series(dtype="float64")

    return pd.to_numeric(series, errors="coerce")


def apply_cleaning(df, categorical_cols=None, numeric_cols=None):
    categorical_cols = categorical_cols or []
    numeric_cols = numeric_cols or []

    for col in categorical_cols:
        df[col] = clean_categorical(df.get(col, pd.Series(index=df.index, dtype="string")))

    for col in numeric_cols:
        df[col] = clean_numerical(df.get(col, pd.Series(index=df.index, dtype="float64")))

    return df
