"""Synthetic checks of the paper specification; no external data files."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from risk_support_decoupling.backward_smoothing import smooth_scores
from risk_support_decoupling.capacity_ranking import rank_and_select
from risk_support_decoupling.config_loader import (
    CATEGORICAL_FEATURES, HIERARCHY_LEVELS, NUMERIC_FEATURES, UNIT_COLUMNS,
    load_reference_config,
)
from risk_support_decoupling.evidence_rates import beta_rate, empirical_prior, log_saturated_support
from risk_support_decoupling.feature_construction import construct_features_for_month, build_supervised_rows
from risk_support_decoupling.hierarchy import adaptive_hierarchy_score
from risk_support_decoupling.monthly_training import fit_or_fallback


@pytest.fixture
def config():
    return deepcopy({
        "prior_strength_a": 100, "support_saturation_n_max": 150,
        "prior_start_month": "2024-01", "local_history_start_month": "2024-07",
        "training_start_month": "2024-07", "latency_months": 0, "default_prior": 0.0,
        "learner": "system_lightgbm", "random_state": 0, "min_training_rows": 1,
        "fallback_score_column": "historical_smoothed_rate",
        "numeric_features": NUMERIC_FEATURES.copy(),
        "categorical_features": CATEGORICAL_FEATURES.copy(),
        "score_smoothing": {"enabled": False, "alpha": 0.6},
        "tie_breaking": [
            {"column": "smoothed_score", "ascending": False},
            {"column": "risk_unit_id", "ascending": True},
        ],
        "missing_category_token": "__missing__", "hierarchy_levels": HIERARCHY_LEVELS,
        "rolling_two_months": 2, "rolling_quarter_months": 3,
        "group_backoff_window_months": None, "include_extra_fields": False,
        "method_status": "reference_implementation_following_paper_specification",
    })


def row(month, n, f, category="kind_a", broad="group_a", stage="stage_a", item="item_a"):
    fields = [broad, category, stage, item]
    return dict(month=month, risk_unit_id="|".join(fields), sample_count=n,
                failure_count=f, **dict(zip(UNIT_COLUMNS, fields)))


def scoring():
    return pd.DataFrame({
        **{field: [0.1, 0.2] for field in NUMERIC_FEATURES},
        "risk_unit_id": ["group_a|kind_a|stage_a|item_a", "group_a|kind_b|stage_a|item_a"],
        "broad_category": ["group_a", "group_a"], "sampling_stage": ["stage_a", "stage_a"],
        "historical_smoothed_rate": [0.2, 0.4], "decision_month": ["2024-10"] * 2,
    })


def training():
    data = pd.concat([scoring()] * 50, ignore_index=True)
    data["decision_month"] = "2024-07"
    data["target_month"] = "2024-08"
    data["target_observed"] = True
    data["target"] = [0, 1] * 50
    return data


@pytest.mark.parametrize("n,expected", [(0, 0), (1, 0.138152), (10, 0.477927), (150, 1), (400, 1)])
def test_support(n, expected):
    assert log_saturated_support(n, 150) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("n,limit", [(-1, 150), (1, 0), (1, -1)])
def test_invalid_support(n, limit):
    with pytest.raises(ValueError):
        log_saturated_support(n, limit)


@pytest.mark.parametrize("f,n,expected", [(1, 2, 0.029412), (0, 50, 0.013333)])
def test_beta_rate(f, n, expected):
    assert beta_rate(f, n, 0.02, 100) == pytest.approx(expected, abs=1e-6)


def test_prior_interval():
    data = pd.DataFrame([row("2023-12", 100, 100), row("2024-01", 100, 2), row("2024-09", 100, 100)])
    assert empirical_prior(data, "2024-08", 0.0, "2024-01") == 0.02
    assert empirical_prior(data, "2023-11", 0.25, "2024-01") == 0.25


def test_zero_sample_prior_is_undefined():
    with pytest.raises(ValueError, match="undefined"):
        empirical_prior(pd.DataFrame([row("2024-01", 0, 0)]), "2024-07", 0.25)


def test_local_history_and_field_order(config):
    data = pd.DataFrame([row("2024-01", 100, 10), row("2024-07", 20, 0)])
    fields = construct_features_for_month(data, "2024-07", config)
    assert fields.columns[:9].tolist() == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert "adaptive_hierarchy_score" not in fields
    assert fields.iloc[0]["cumulative_sample_count"] == 20
    assert fields.iloc[0]["empirical_prior"] == pytest.approx(10 / 120)
    config["include_extra_fields"] = True
    extra = construct_features_for_month(data, "2024-07", config)
    assert {"level_0_rate", "adaptive_hierarchy_score", "selected_failure_count"} <= set(extra)


def test_rebuild_prior_at_each_decision(config):
    data = pd.DataFrame([row("2024-07", 10, 1), row("2024-08", 10, 5), row("2024-09", 10, 0)])
    rows = build_supervised_rows(data, ["2024-06", "2024-07", "2024-08"], config)
    assert rows["decision_month"].tolist() == ["2024-07", "2024-08"]
    assert rows["empirical_prior"].tolist() == pytest.approx([0.1, 0.3])


def test_latency_and_training_boundary(config):
    config["latency_months"] = 1
    data = pd.DataFrame([row("2024-06", 10, 1), row("2024-07", 10, 1), row("2024-08", 10, 5), row("2024-09", 100, 100)])
    fields = construct_features_for_month(data, "2024-09", config)
    assert fields.iloc[0]["history_cutoff"] == "2024-08"
    assert fields.iloc[0]["cumulative_sample_count"] == 20
    assert fields.iloc[0]["month_of_year"] == 9
    rows = build_supervised_rows(data, ["2024-08"], config)
    assert rows.iloc[0]["empirical_prior"] == 0.1
    assert rows.iloc[0]["target"] == 1
    with pytest.raises(ValueError, match="boundary"):
        fit_or_fallback(rows, fields, "2024-09", config)
    earlier = build_supervised_rows(data, ["2024-07"], config)
    assert not earlier.empty
    assert (earlier["decision_month"].map(pd.Period) + 1 <= pd.Period("2024-08")).all()
    fit_or_fallback(earlier, fields, "2024-09", config)


def test_unobserved_labels_and_invalid_targets(config, monkeypatch):
    data = pd.DataFrame([row("2024-07", 10, 1), row("2024-07", 10, 0, "kind_b"), row("2024-08", 10, 1)])
    rows = build_supervised_rows(data, ["2024-07"], config)
    missing = rows.loc[~rows["target_observed"]]
    assert len(missing) == 1 and missing["target"].isna().all()
    def no_fit(*args, **kwargs):
        pytest.fail("Unobserved rows must not supply the second training class.")
    monkeypatch.setattr(Pipeline, "fit", no_fit)
    result = fit_or_fallback(rows, scoring(), "2024-10", config)
    assert (result["observed_training_rows"] == 1).all()
    assert (result["model_mode"] == "fallback_single_class").all()
    invalid = training().iloc[:2].copy()
    invalid["target"] = [0, 2]
    assert (fit_or_fallback(invalid, scoring(), "2024-10", config)["observed_training_rows"] == 1).all()


@pytest.mark.parametrize("field", ["category", "tested_item", "risk_unit_id", "enterprise_id", "company_name"])
def test_forbidden_features(config, field):
    config["numeric_features"].append(field)
    with pytest.raises(ValueError, match="not permitted"):
        fit_or_fallback(pd.DataFrame(), scoring(), "2024-10", config)


def test_system_parameters(config, monkeypatch):
    captured = []
    original = Pipeline.fit
    def capture(self, *args, **kwargs):
        if "classifier" in self.named_steps:
            captured.append(self)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Pipeline, "fit", capture)
    scores = scoring()
    scores["broad_category"] = "unseen_group"
    result = fit_or_fallback(training(), scores, "2024-10", config)
    params = captured[0].named_steps["classifier"].get_params()
    expected = dict(objective="binary", n_estimators=200, num_leaves=7, max_depth=3,
                    learning_rate=0.03, min_child_samples=40, reg_lambda=2.0, n_jobs=4,
                    deterministic=True, force_row_wise=True, random_state=0, verbose=-1)
    assert all(params[key] == value for key, value in expected.items())
    assert result["raw_score"].between(0, 1).all()
    categories = captured[0].named_steps["preprocess"].named_transformers_["categorical"].named_steps["one_hot"].categories_
    assert "unseen_group" not in categories[0]


def test_sklearn_parameters_and_override(config, monkeypatch):
    config["learner"] = "gbm9_sklearn"
    captured = []
    original = Pipeline.fit
    def capture(self, *args, **kwargs):
        if "classifier" in self.named_steps:
            captured.append(self.named_steps["classifier"])
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Pipeline, "fit", capture)
    fit_or_fallback(training(), scoring(), "2024-10", config)
    params = captured[-1].get_params()
    assert (params["n_estimators"], params["max_depth"], params["learning_rate"]) == (80, 2, 0.05)
    config["learner_params"] = {"n_estimators": 2}
    fit_or_fallback(training(), scoring(), "2024-10", config)
    assert captured[-1].get_params()["n_estimators"] == 2


def test_ranking_ties(config):
    scores = pd.DataFrame({"risk_unit_id": ["b", "a", "c"], "raw_score": [0.5] * 3})
    result = rank_and_select(scores, 1, {})
    assert result.loc[result["selected"], "risk_unit_id"].tolist() == ["a"]
    assert result["rank"].tolist() == [1, 2, 3]
    assert not rank_and_select(scores, 0, {})["selected"].any()


def test_empty_and_single_class_fallback(config):
    scores = scoring()
    for train in [pd.DataFrame(), training().assign(target=1)]:
        result = fit_or_fallback(train, scores, "2024-10", config)
        np.testing.assert_array_equal(result["raw_score"], scores["historical_smoothed_rate"])


def test_hierarchy_two_months(config):
    first = row("2024-07", 7, 1)
    data = pd.DataFrame([first, row("2024-08", 5, 0)])
    value = adaptive_hierarchy_score(data, first, "2024-08", config)
    assert value.selected_level_index == 1 and value.selected_window_months == 2
    assert value.support_count == 12 and value.failure_count == 1
    assert value.score == pytest.approx(beta_rate(1, 12, 1 / 12, 100))


def test_hierarchy_group_backoff(config):
    first = row("2024-07", 5, 0)
    data = pd.DataFrame([first, row("2024-07", 55, 0, "kind_b")])
    value = adaptive_hierarchy_score(data, first, "2024-07", config)
    assert value.selected_level == "broad_category+sampling_stage+tested_item"
    assert value.selected_level_index == 3 and value.support_count == 60
    assert value.selected_window_months is None
    config["group_backoff_window_months"] = 1
    value = adaptive_hierarchy_score(data, first, "2024-08", config)
    assert value.selected_level == "global_prior"


@pytest.mark.parametrize("other,index", [
    ({"category": "kind_b"}, 3),
    ({"item": "item_b"}, 4),
    ({"category": "kind_b", "item": "item_b"}, 5),
    ({"stage": "stage_b", "category": "kind_b", "item": "item_b"}, 6),
])
def test_all_group_sources_in_order(config, other, index):
    first = row("2024-07", 5, 0)
    data = pd.DataFrame([first, row("2024-07", 60, 0, **other)])
    value = adaptive_hierarchy_score(data, first, "2024-07", config)
    assert value.selected_level_index == index
    assert value.support_count == 65


def test_third_unit_window(config):
    first = row("2024-07", 40, 0)
    data = pd.DataFrame([first, row("2024-08", 5, 0), row("2024-09", 5, 0)])
    value = adaptive_hierarchy_score(data, first, "2024-09", config)
    assert value.selected_level_index == 2
    assert value.selected_window_months == 3 and value.support_count == 50


def test_training_start_and_gbm11_fallback(config):
    config.update(learner="gbm9_sklearn", min_training_rows=20,
                  fallback_score_column="adaptive_hierarchy_score", include_extra_fields=True)
    config["numeric_features"] += ["level_0_rate", "adaptive_hierarchy_score"]
    scores = scoring().assign(adaptive_hierarchy_score=[0.7, 0.8])
    early = training().iloc[:10].copy()
    early["decision_month"] = "2024-06"
    early["target_month"] = "2024-07"
    result = fit_or_fallback(early, scores, "2024-10", config)
    assert (result["observed_training_rows"] == 0).all()
    np.testing.assert_array_equal(result["raw_score"], scores["adaptive_hierarchy_score"])
    result = fit_or_fallback(training().iloc[:10], scores, "2024-10", config)
    assert (result["observed_training_rows"] == 10).all()
    np.testing.assert_array_equal(result["raw_score"], scores["adaptive_hierarchy_score"])


def test_smoothing_switch(config):
    current = scoring().assign(raw_score=[0.2, 0.4])
    previous = current.assign(decision_month="2024-09", smoothed_score=[0.8, 0.1])
    disabled = smooth_scores(current, config, previous_scores=previous)
    np.testing.assert_array_equal(disabled["smoothed_score"], current["raw_score"])
    assert (disabled["smoothing_mode"] == "disabled").all()
    config["score_smoothing"]["enabled"] = True
    result = smooth_scores(current, config, previous_scores=previous.iloc[:1])
    assert result["smoothed_score"].tolist() == pytest.approx([0.44, 0.4])
    with pytest.raises(ValueError, match="earlier"):
        smooth_scores(current, config, previous_scores=previous.assign(decision_month="2024-10"))


def test_four_field_identity(config):
    data = pd.DataFrame([row("2024-07", 10, 1), row("2024-08", 10, 0)])
    data.loc[1, "risk_unit_id"] = "different_key"
    with pytest.raises(ValueError, match="one-to-one"):
        construct_features_for_month(data, "2024-08", config)


@pytest.mark.parametrize("key,value", [("latency_months", 2), ("prior_strength_a", 0), ("support_saturation_n_max", 0), ("learner", "unknown")])
def test_config_validation_without_file_io(config, monkeypatch, key, value):
    from contextlib import nullcontext
    from pathlib import Path
    config[key] = value
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: nullcontext(None))
    monkeypatch.setattr("risk_support_decoupling.config_loader.yaml.safe_load", lambda _: config)
    with pytest.raises(ValueError):
        load_reference_config("unused.yaml")
