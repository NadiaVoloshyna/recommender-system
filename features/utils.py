def validate_columns(df, required_columns, name):
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )

