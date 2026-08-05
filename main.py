from etl.pipeline import run_etl_pipeline
# from features.pipeline import build_features


def main():
    data = run_etl_pipeline(fetch_api_data=False)
    # features = build_features(data)


if __name__ == "__main__":
    main()

