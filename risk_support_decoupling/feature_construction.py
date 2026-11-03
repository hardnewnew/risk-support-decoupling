"""Candidate reference implementation; not yet verified as identical to the implementation used for the reported results."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from .evidence_rates import aggregate_counts, as_month, beta_rate, empirical_prior, restrict_to_cutoff, validate_aggregates
from .hierarchy import adaptive_hierarchy_score


def _normalize_input(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Validate unit identifiers and normalize configured categorical keys."""

    normalized = validate_aggregates(data)
    if "risk_unit_id" not in normalized.columns:
        raise ValueError("Aggregate data require risk_unit_id.")
    if normalized["risk_unit_id"].isna().any():
        raise ValueError("risk_unit_id cannot be missing.")
    normalized["risk_unit_id"] = normalized["risk_unit_id"].astype(str)
    token = str(config["missing_category_token"])
    configured_columns = {
        column
        for level in config["hierarchy_levels"]
        for column in level
    }.union(config["categorical_features"])
    for column in sorted(configured_columns):
        if column not in normalized.columns:
            normalized[column] = token
        normalized[column] = normalized[column].where(normalized[column].notna(), token).astype(str)
    return normalized


def _candidate_table(
    history: pd.DataFrame,
    config: dict[str, Any],
    candidates: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build one deterministic candidate row per unit from cutoff-available attributes."""

    token = str(config["missing_category_token"])
    attribute_columns = sorted(
        {
            column
            for level in config["hierarchy_levels"]
            for column in level
        }.union(config["categorical_features"])
        - {"risk_unit_id"}
    )
    if candidates is None:
        if history.empty:
            return pd.DataFrame(columns=["risk_unit_id", *attribute_columns])
        ordered = history.sort_values(["month", "risk_unit_id"], kind="mergesort")
        result = ordered.drop_duplicates("risk_unit_id", keep="last")
        return result[["risk_unit_id", *attribute_columns]].reset_index(drop=True)

    if "risk_unit_id" not in candidates.columns or candidates["risk_unit_id"].isna().any():
        raise ValueError("Explicit candidates require nonmissing risk_unit_id.")
    result = candidates.copy()
    result["risk_unit_id"] = result["risk_unit_id"].astype(str)
    if result["risk_unit_id"].duplicated().any():
        raise ValueError("Explicit candidates must be unique by risk_unit_id.")
    for column in attribute_columns:
        if column not in result.columns:
            result[column] = token
        result[column] = result[column].where(result[column].notna(), token).astype(str)
    return result[["risk_unit_id", *attribute_columns]].reset_index(drop=True)


def construct_features_for_month(
    data: pd.DataFrame,
    decision_month: Any,
    config: dict[str, Any],
    *,
    candidates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construct the 11 proposed forward-safe features for one decision month."""

    normalized = _normalize_input(data, config)
    cutoff = as_month(decision_month)
    history = restrict_to_cutoff(normalized, cutoff)
    if not history.empty and history["month"].max() > cutoff:
        raise AssertionError("Feature history exceeds the decision month.")
    candidate_table = _candidate_table(history, config, candidates)
    prior = empirical_prior(history, cutoff, float(config["default_prior"]))
    m0 = float(config["m0"])
    tau = float(config["tau"])
    rows: list[dict[str, Any]] = []

    for candidate in candidate_table.to_dict(orient="records"):
        unit_id = str(candidate["risk_unit_id"])
        unit_history = history.loc[history["risk_unit_id"] == unit_id]
        cumulative_n, cumulative_f = aggregate_counts(unit_history, cutoff)
        two_n, two_f = aggregate_counts(
            unit_history, cutoff, window_months=int(config["rolling_two_months"])
        )
        fixed_n, fixed_f = aggregate_counts(
            unit_history, cutoff, window_months=int(config["fixed_window_months"])
        )
        quarter_n, _ = aggregate_counts(
            unit_history, cutoff, window_months=int(config["rolling_quarter_months"])
        )
        hierarchy = adaptive_hierarchy_score(history, candidate, cutoff, config)
        level_zero_rate = cumulative_f / cumulative_n if cumulative_n > 0 else prior
        support_score = cumulative_n / (cumulative_n + tau) if cumulative_n > 0 else 0.0
        row = {
            "risk_unit_id": unit_id,
            "decision_month": str(cutoff),
            "history_cutoff": str(cutoff),
            "historical_smoothed_rate": beta_rate(cumulative_f, cumulative_n, prior, m0),
            "rolling_two_month_beta_rate": beta_rate(two_f, two_n, prior, m0),
            "fixed_window_beta_rate": beta_rate(fixed_f, fixed_n, prior, m0),
            "level_0_rate": float(level_zero_rate),
            "adaptive_hierarchy_score": hierarchy.score,
            "support_score": float(support_score),
            "cumulative_sample_count": float(cumulative_n),
            "rolling_quarter_sample_count": float(quarter_n),
            "month_of_year": int(cutoff.month),
            "broad_category": candidate.get("broad_category", config["missing_category_token"]),
            "sampling_stage": candidate.get("sampling_stage", config["missing_category_token"]),
            "selected_hierarchy_level": hierarchy.selected_level,
            "selected_hierarchy_level_index": hierarchy.selected_level_index,
            "selected_window_months": hierarchy.selected_window_months,
            "selected_support_count": hierarchy.support_count,
            "empirical_prior": prior,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_supervised_rows(
    data: pd.DataFrame,
    decision_months: Iterable[Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build earlier decision rows with masked, next-month observed labels."""

    normalized = _normalize_input(data, config)
    panels: list[pd.DataFrame] = []
    for value in decision_months:
        decision = as_month(value)
        target_month = decision + 1
        features = construct_features_for_month(normalized, decision, config)
        if features.empty:
            continue
        target = normalized.loc[normalized["month"] == target_month]
        target_counts = (
            target.groupby("risk_unit_id", as_index=False)[["sample_count", "failure_count"]].sum()
            if not target.empty
            else pd.DataFrame(columns=["risk_unit_id", "sample_count", "failure_count"])
        )
        target_map = target_counts.set_index("risk_unit_id").to_dict(orient="index")
        observed: list[bool] = []
        labels: list[float] = []
        for unit_id in features["risk_unit_id"]:
            counts = target_map.get(str(unit_id))
            is_observed = bool(counts is not None and counts["sample_count"] > 0)
            observed.append(is_observed)
            labels.append(float(counts["failure_count"] > 0) if is_observed else float("nan"))
        features["target_month"] = str(target_month)
        features["target_observed"] = observed
        features["target"] = labels
        panels.append(features)
    if not panels:
        return pd.DataFrame()
    return pd.concat(panels, ignore_index=True)

