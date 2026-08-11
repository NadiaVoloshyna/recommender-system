import pandas as pd


def validate_columns(df: pd.DataFrame, required_columns: list[str], name: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {', '.join(missing)}"
        )
