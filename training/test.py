from training.negative_sampling import add_candidate_hardness_scores, HARDNESS_FEATURES, sample_negatives_by_hardness
from training.preprocessing import fit_transform_features, transform_features, COLUMNS_TO_DROP, COLUMNS_TO_SCALE
from training.evaluation import precision_at_k, recall_at_k, ndcg_at_k
from training.train_baseline import train_baseline
import training.train_nn as train_nn_module
from training.train_nn import train_nn, NN, evaluate
from training.train_gbm import train_gbm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_candidate_df():
    df = pd.DataFrame({
        "user_id": ["A", "A", "A", "B", "B", "B"],
        "source_interaction_strength_log": [
            1.0, 2.0, 3.0,
            10.0, 20.0, 30.0],
        "vector_similarity_score": [1, 2, 3, 1, 2, 3],
        "track_similarity_score": [1, 2, 3, 1, 2, 3],
        "artist_similarity_score": [1, 2, 3, 1, 2, 3],
        "candidate_relative_global_popularity": [1, 2, 3, 1, 2, 3]})

    return df


def make_full_candidate_df():
    df = pd.DataFrame({
        "user_id": ["A", "A", "A", "B", "B", "B"],
        "track_id": ["T1", "T2", "T3", "T1", "T2", "T3"],
        "source_interaction_strength": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        "source_interaction_strength_log": [
            1.0, 2.0, 3.0,
            10.0, 20.0, 30.0],
        "n_sources": [1, 2, 3, 1, 2, 3],
        "track_similarity_score": [1, 2, 3, 1, 2, 3],
        "artist_similarity_score": [1, 2, 3, 1, 2, 3],
        "vector_similarity_score": [1, 2, 3, 1, 2, 3],
        "max_similarity": [1, 2, 3, 1, 2, 3],
        "mean_similarity_available": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "mean_similarity_all": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "track_interaction_signal": [1.0, 4.0, 9.0, 10.0, 40.0, 90.0],
        "artist_interaction_signal": [1.0, 4.0, 9.0, 10.0, 40.0, 90.0],
        "vector_interaction_signal": [1.0, 4.0, 9.0, 10.0, 40.0, 90.0],
        "global_popularity": [1, 2, 3, 1, 2, 3],
        "global_popularity_log": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "candidate_relative_global_popularity": [1, 2, 3, 1, 2, 3],
        "global_popularity_missing": [0, 0, 0, 0, 0, 0],
        "track_similarity_available": [1, 1, 1, 1, 1, 1],
        "artist_similarity_available": [1, 1, 1, 1, 1, 1],
        "vector_similarity_available": [1, 1, 1, 1, 1, 1],
        "label": [0, 1, 0, 1, 0, 1]})

    return df


# add_candidate_hardness_scores() adds all expected columns
def test_add_candidate_hardness_scores_adds_expected_columns():
    df = make_candidate_df()

    result = add_candidate_hardness_scores(df)

    for feature in HARDNESS_FEATURES:
        assert f"{feature}_rank" in result.columns

    assert "hardness_score" in result.columns
    assert "hardness_percentile" in result.columns
    assert "hardness_bucket" in result.columns


# add_candidate_hardness_scores() does not mutate input
def test_add_candidate_hardness_scores_does_not_modify_input():
    df = make_candidate_df()
    original = df.copy(deep=True)

    add_candidate_hardness_scores(df)

    pd.testing.assert_frame_equal(df, original)


# add_candidate_hardness_scores() ranks independently per user
def test_add_candidate_hardness_scores_percentile_ranks_are_calculated_per_user():
    df = make_candidate_df()

    result = add_candidate_hardness_scores(df)

    expected = [1 / 3, 2 / 3, 1.0] * 2

    actual = result["source_interaction_strength_log_rank"]
    np.testing.assert_allclose(actual, expected)


# add_candidate_hardness_scores() Calculates weighted hardness correctly
def test_add_candidate_hardness_scores_uses_configured_weights():
    df = make_candidate_df()

    result = add_candidate_hardness_scores(df)

    expected = sum(
        weight * result[f"{feature}_rank"]
        for feature, weight in HARDNESS_FEATURES.items())

    pd.testing.assert_series_equal(
        result["hardness_score"],
        expected,
        check_names=False)


# add_candidate_hardness_scores() Assigns valid hardness buckets
def test_add_candidate_hardness_scores_bucket_values_are_valid():
    df = make_candidate_df()
    result = add_candidate_hardness_scores(df)
    valid_buckets = {"easy", "medium", "hard", "very_hard"}

    assert set(
        result["hardness_bucket"].dropna().astype(str)
    ).issubset(valid_buckets)


BUCKETS = ["easy", "medium", "hard", "very_hard"]


def make_test_df(
    positives_per_user: dict[str, int],
    negatives_per_user: int,
    buckets: list[str] | None = None,
) -> pd.DataFrame:
    """
    Creates a synthetic candidate DataFrame. Each user receives: the requested number of positives,
    the requested number of negatives. Negative candidates are distributed across hardness buckets
    in round-robin fashion unless `buckets` is supplied.
    """
    if buckets is None:
        buckets = BUCKETS

    rows = []
    row_id = 1000

    for user_id, n_positives in positives_per_user.items():
        for i in range(n_positives):
            rows.append(
                {
                    "row_id": row_id,
                    "user_id": user_id,
                    "track_id": f"{user_id}_positive_{i}",
                    "label": 1,
                    "hardness_bucket": None})
            row_id += 1

        for i in range(negatives_per_user):
            rows.append(
                {
                    "row_id": row_id,
                    "user_id": user_id,
                    "track_id": f"{user_id}_negative_{i}",
                    "label": 0,
                    "hardness_bucket": buckets[i % len(buckets)]})
            row_id += 1

    return pd.DataFrame(rows).set_index("row_id")


def make_balanced_bucket_df(
    n_positives: int = 1,
    negatives_per_bucket: int = 10,
) -> pd.DataFrame:
    """
    Creates one user's candidate pool with an equal number of negatives in every hardness bucket.
    """
    rows = []
    row_id = 1000

    for i in range(n_positives):
        rows.append(
            {
                "row_id": row_id,
                "user_id": "user_1",
                "track_id": f"positive_{i}",
                "label": 1,
                "hardness_bucket": None})
        row_id += 1

    for bucket in BUCKETS:
        for i in range(negatives_per_bucket):
            rows.append(
                {
                    "row_id": row_id,
                    "user_id": "user_1",
                    "track_id": f"{bucket}_{i}",
                    "label": 0,
                    "hardness_bucket": bucket})
            row_id += 1

    return pd.DataFrame(rows).set_index("row_id")


# sample_negatives_by_hardness() keeps all positives
def test_sample_negatives_by_hardness_keeps_all_positives():
    df = make_test_df(
        positives_per_user={"user_1": 3},
        negatives_per_user=100)
    expected_positive_ids = set(df.loc[df["label"] == 1, "track_id"])

    result = sample_negatives_by_hardness(
        df,
        negatives_per_positive=10,
        random_state=42)

    actual_positive_ids = set(result.loc[result["label"] == 1, "track_id"])

    assert actual_positive_ids == expected_positive_ids


# sample_negatives_by_hardness() samples correct number of negatives
def test_sample_negatives_by_hardness_samples_correct_number_of_negatives():
    df = make_test_df(
        positives_per_user={"user_1": 4},
        negatives_per_user=100)

    result = sample_negatives_by_hardness(
        df,
        negatives_per_positive=10,
        random_state=42)

    n_positives = (result["label"] == 1).sum()
    n_negatives = (result["label"] == 0).sum()

    assert n_positives == 4
    assert n_negatives == 40


# sample_negatives_by_hardness(): sampling is independent per user
def test_sample_negatives_by_hardness_sampling_is_independent_per_user():
    df = make_test_df(
        positives_per_user={
            "user_1": 2,
            "user_2": 5,
        },
        negatives_per_user=100)

    result = sample_negatives_by_hardness(
        df,
        negatives_per_positive=10,
        random_state=42)

    for user_id, user_result in result.groupby("user_id"):
        n_positives = (user_result["label"] == 1).sum()
        n_negatives = (user_result["label"] == 0).sum()

        assert n_negatives == (n_positives * 10)


# sample_negatives_by_hardness() handles missing bucket. shortfall
def test_missing_hardness_bucket_fills_shortfall():
    """
    There are no very_hard negatives. The function should still produce the requested
    number of negatives by sampling from the remaining pool.
    """
    df = make_test_df(
        positives_per_user={"user_1": 1},
        negatives_per_user=40,
        buckets=[
            "easy",
            "medium",
            "hard"])
    result = sample_negatives_by_hardness(
        df,
        negatives_per_positive=10,
        random_state=42)

    n_negatives = (result["label"] == 0).sum()

    assert n_negatives == 10


# fit_transform_features() returns correct types
def test_fit_transform_features_returns_correct_types():
    features = make_full_candidate_df()

    X_train, scaler = fit_transform_features(features)

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(scaler, StandardScaler)


# fit_transform_features() removes unwanted columns
def test_fit_transform_features_removes_unwanted_columns():
    features = make_full_candidate_df()

    X_train, scaler = fit_transform_features(features)

    for column in COLUMNS_TO_DROP:
        assert column not in X_train.columns


# fit_transform_features() standardizes selected columns
def test_fit_transform_features_standardizes_selected_columns():
    features = make_full_candidate_df()

    X_train, scaler = fit_transform_features(features)

    for column in COLUMNS_TO_SCALE:
        assert X_train[column].mean() == pytest.approx(0)
        assert X_train[column].std(ddof=0) == pytest.approx(1)


# transform_features() removes unwanted columns
def test_transform_features_removes_unwanted_columns():
    features = make_full_candidate_df()
    scaler = StandardScaler()
    scaler.fit(features[COLUMNS_TO_SCALE])

    X = transform_features(features, scaler)

    for column in COLUMNS_TO_DROP:
        assert column not in X.columns


# transform_features()  uses fitted scaler
def test_transform_features_uses_fitted_scaler():
    features = make_full_candidate_df()
    scaler = StandardScaler()
    scaler.fit(features[COLUMNS_TO_SCALE])

    X = transform_features(features, scaler)

    expected = scaler.transform(features[COLUMNS_TO_SCALE])

    pd.testing.assert_frame_equal(
        X[COLUMNS_TO_SCALE].reset_index(drop=True),
        pd.DataFrame(expected, columns=COLUMNS_TO_SCALE))


# precision_at_k() calculates correctly
def test_precision_at_k_calculates_correctly():
    df = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 2],
        "track_id": [10, 20, 30, 40, 50, 60],
        "label": [1, 0, 1, 0, 1, 1],
        "score": [0.9, 0.8, 0.7, 0.95, 0.85, 0.75]})

    result = precision_at_k(df, k=2)

    assert result == pytest.approx(0.5)


# recall_at_k() calculates correctly
def test_recall_at_k_calculates_correctly():
    df = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 2],
        "track_id": [10, 20, 30, 40, 50, 60],
        "label": [1, 1, 0, 1, 0, 0],
        "score": [0.9, 0.8, 0.7, 0.95, 0.85, 0.75]})

    result = recall_at_k(df, k=2)

    assert result == pytest.approx(1.0)


# recall_at_k() ignores_users_with_no_relevant_tracks
def test_recall_at_k_ignores_users_with_no_relevant_tracks():
    df = pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "track_id": [10, 20, 30, 40],
        "label": [1, 0, 0, 0],
        "score": [0.9, 0.8, 0.95, 0.85]})

    result = recall_at_k(df, k=1)

    assert result == pytest.approx(1.0)


# ndcg_at_k() calculates correctly
def test_ndcg_at_k_calculates_correctly():
    df = pd.DataFrame({
        "user_id": [1, 1, 1],
        "track_id": [10, 20, 30],
        "label": [1, 0, 1],
        "score": [0.9, 0.8, 0.7]})

    result = ndcg_at_k(df, k=2)

    expected = 1 / (1 + 1 / np.log2(3))

    assert result == pytest.approx(expected)


# ndcg_at_k() ignores users with no relevant tracks
def test_ndcg_at_k_ignores_users_with_no_relevant_tracks():
    df = pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "track_id": [10, 20, 30, 40],
        "label": [1, 0, 0, 0],
        "score": [0.9, 0.8, 0.95, 0.85]})

    result = ndcg_at_k(df, k=2)

    assert result == pytest.approx(1.0)


@pytest.fixture
def train_data():
    return pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "track_id": ["a", "b", "a", "b"],
        "source_interaction_strength": [0.1, 0.2, 0.3, 0.4],
        "global_popularity": [1.0, 2.0, 3.0, 4.0],
        "source_interaction_strength_log": [0.1, 0.2, 0.3, 0.4],
        "n_sources": [1, 2, 1, 2],
        "track_similarity_score": [0.2, 0.3, 0.4, 0.5],
        "artist_similarity_score": [0.3, 0.4, 0.5, 0.6],
        "vector_similarity_score": [0.4, 0.5, 0.6, 0.7],
        "max_similarity": [0.4, 0.5, 0.6, 0.7],
        "mean_similarity_available": [0.3, 0.4, 0.5, 0.6],
        "mean_similarity_all": [0.3, 0.4, 0.5, 0.6],
        "track_interaction_signal": [0.1, 0.2, 0.3, 0.4],
        "artist_interaction_signal": [0.1, 0.2, 0.3, 0.4],
        "vector_interaction_signal": [0.1, 0.2, 0.3, 0.4],
        "global_popularity_log": [1.0, 2.0, 3.0, 4.0],
        "candidate_relative_global_popularity": [0.1, 0.2, 0.3, 0.4],
        "label": [1, 0, 0, 1]})


@pytest.fixture
def val_data():
    return pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "track_id": ["a", "b", "a", "b"],
        "source_interaction_strength": [0.15, 0.25, 0.35, 0.45],
        "global_popularity": [1.5, 2.5, 3.5, 4.5],
        "source_interaction_strength_log": [0.15, 0.25, 0.35, 0.45],
        "n_sources": [1, 2, 1, 2],
        "track_similarity_score": [0.25, 0.35, 0.45, 0.55],
        "artist_similarity_score": [0.35, 0.45, 0.55, 0.65],
        "vector_similarity_score": [0.45, 0.55, 0.65, 0.75],
        "max_similarity": [0.45, 0.55, 0.65, 0.75],
        "mean_similarity_available": [0.35, 0.45, 0.55, 0.65],
        "mean_similarity_all": [0.35, 0.45, 0.55, 0.65],
        "track_interaction_signal": [0.15, 0.25, 0.35, 0.45],
        "artist_interaction_signal": [0.15, 0.25, 0.35, 0.45],
        "vector_interaction_signal": [0.15, 0.25, 0.35, 0.45],
        "global_popularity_log": [1.5, 2.5, 3.5, 4.5],
        "candidate_relative_global_popularity": [0.15, 0.25, 0.35, 0.45],
        "label": [1, 0, 0, 1]})


# train_baseline() rejects invalid input
def test_train_baseline_rejects_invalid_input(train_data, val_data):
    with pytest.raises(TypeError):
        train_baseline(
            train_data.to_numpy(),
            train_data,
            val_data)

    with pytest.raises(TypeError):
        train_baseline(
            train_data,
            train_data.to_numpy(),
            val_data)

    with pytest.raises(TypeError):
        train_baseline(
            train_data,
            train_data,
            val_data.to_numpy())


# train_baseline() returns expected objects
def test_train_baseline_returns_expected_objects(train_data, val_data, monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    model, scaler, metrics, selected_source = train_baseline(
        train_data,
        train_data,
        val_data)

    assert isinstance(model, LogisticRegression)
    assert isinstance(scaler, StandardScaler)
    assert isinstance(metrics, dict)
    assert isinstance(selected_source, str)


# train_baseline() returns all metrics
def test_train_baseline_returns_all_metrics(train_data, val_data, monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    _, _, metrics, selected_model = train_baseline(
        train_data,
        train_data,
        val_data)

    expected_metrics = {
        "auc",
        "precision@10",
        "recall@10",
        "ndcg@10",
        "precision@20",
        "recall@20",
        "ndcg@20"}

    assert selected_model in {"FULL", "SAMPLED"}
    assert set(metrics.keys()) == expected_metrics


@pytest.fixture
def scaler(train_data):
    scaler = StandardScaler()
    scaler.fit(train_data[COLUMNS_TO_SCALE])
    return scaler


# train_nn() returns the correct outputs
def test_train_nn_returns_model_and_metrics(
    train_data,
    val_data,
    scaler,
    monkeypatch
):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    model, metrics, _, _, _ = train_nn(train_data, val_data, scaler)

    assert isinstance(model, NN)
    assert isinstance(metrics, dict)


# train_nn() returns expected metrics
def test_train_nn_returns_expected_metrics(
    train_data,
    val_data,
    scaler,
    monkeypatch
):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    model, metrics, _, _, _ = train_nn(train_data, val_data, scaler)

    expected_metrics = {
        "auc",
        "precision@10",
        "recall@10",
        "ndcg@10",
        "precision@20",
        "recall@20",
        "ndcg@20",
    }

    assert expected_metrics.issubset(metrics.keys())


# train_nn() selects best ndcg10
def test_train_nn_selects_best_ndcg10(
    train_data,
    val_data,
    scaler,
    monkeypatch
):
    monkeypatch.setattr(train_nn_module, "EPOCHS", 3)
    monkeypatch.setattr(
        train_nn_module,
        "plot_training_history",
        lambda history: None)

    ndcg_values = [0.30, 0.50, 0.40]

    def mock_evaluate(*args, **kwargs):
        ndcg = ndcg_values.pop(0)
        metrics = {
            "auc": 0.80,
            "precision@10": 0.50,
            "recall@10": 0.40,
            "ndcg@10": ndcg,
            "precision@20": 0.45,
            "recall@20": 0.50,
            "ndcg@20": 0.45,
        }
        return 0.5, metrics

    monkeypatch.setattr(
        train_nn_module,
        "evaluate",
        mock_evaluate)

    model, metrics, _, _, _ = train_nn(train_data, val_data, scaler)

    assert isinstance(model, NN)
    # Epoch 2 had the highest NDCG@10.
    assert metrics["ndcg@10"] == 0.50


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def validation_data():
    features = pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "track_id": [10, 11, 10, 12],
        "label": [1, 0, 1, 0],
    })

    numeric = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ])

    targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

    loader = DataLoader(
        TensorDataset(numeric, targets),
        batch_size=2,
        shuffle=False,
    )

    return features, loader


@pytest.fixture
def model():
    return DummyModel()


@pytest.fixture
def criterion():
    return nn.BCEWithLogitsLoss()


# evaluate() returns loss and metrics
def test_evaluate_returns_loss_and_metrics(model, validation_data, criterion):
    val_features, loader = validation_data

    loss, metrics = evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=torch.device("cpu"),
        val_features=val_features,
    )

    assert isinstance(loss, float)
    assert loss >= 0

    assert set(metrics) == {
        "auc",
        "precision@10",
        "recall@10",
        "ndcg@10",
        "precision@20",
        "recall@20",
        "ndcg@20"}

    assert all(isinstance(value, float) for value in metrics.values())


# evaluate() does not update model
def test_evaluate_does_not_update_model(model, validation_data, criterion):
    val_features, loader = validation_data

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=torch.device("cpu"),
        val_features=val_features)

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])


# evaluate() rejects single class targets
def test_evaluate_rejects_single_class_targets(model, criterion):
    numeric = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0]])
    targets = torch.tensor([[1.0], [1.0]])

    loader = DataLoader(
        TensorDataset(numeric, targets),
        batch_size=2,
        shuffle=False)

    val_features = pd.DataFrame({
        "user_id": [1, 1],
        "track_id": [10, 11],
        "label": [1, 1]})

    with pytest.raises(ValueError, match="both classes"):
        evaluate(
            model=model,
            loader=loader,
            criterion=criterion,
            device=torch.device("cpu"),
            val_features=val_features)


# train_gbm() is trained
def test_train_gbm_returns_model(train_data, val_data):
    model, metrics = train_gbm(train_data, val_data)

    assert isinstance(model, LGBMClassifier)


# train_gbm() returns metrics
def test_train_gbm_returns_metrics(train_data, val_data):
    _, metrics = train_gbm(train_data, val_data)

    expected = {
        "auc",
        "precision@10",
        "recall@10",
        "ndcg@10",
        "precision@20",
        "recall@20",
        "ndcg@20",
    }

    assert set(metrics.keys()) == expected


# train_gbm() returns valid metrics
def test_train_gbm_metrics_are_valid(train_data, val_data):
    _, metrics = train_gbm(train_data, val_data)

    assert 0 <= metrics["auc"] <= 1

    for metric in [
        "precision@10",
        "recall@10",
        "ndcg@10",
        "precision@20",
        "recall@20",
        "ndcg@20",
    ]:
        assert 0 <= metrics[metric] <= 1

