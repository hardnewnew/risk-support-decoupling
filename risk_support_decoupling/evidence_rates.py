"""Reference implementation following the method specification of the accompanying paper; see README for scope."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from math import isfinite, log1p

import pandas as pd


def as_month(value: Any) -> pd.Period:
    """Convert a supported month value to a monthly pandas Period."""

    return value if isinstance(value, pd.Period) else pd.Period(str(value), freq="M")


def validate_aggregates(data: pd.DataFrame) -> pd.DataFrame:
    """Validate monthly aggregate counts and return a normalized copy."""

    required = {"month", "sample_count", "failure_count"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Aggregate data are missing required columns: {missing}")
    if data[list(required)].isna().any().any():
        raise ValueError("Month and count fields cannot be missing.")
    normalized = data.copy()
    normalized["month"] = normalized["month"].map(as_month)
    normalized["sample_count"] = pd.to_numeric(normalized["sample_count"], errors="raise")
    normalized["failure_count"] = pd.to_numeric(normalized["failure_count"], errors="raise")
    invalid = (
        ~normalized["sample_count"].map(isfinite)
        | ~normalized["failure_count"].map(isfinite)
        | (normalized["sample_count"] < 0)
        | (normalized["failure_count"] < 0)
        | (normalized["failure_count"] > normalized["sample_count"])
    )
    if invalid.any():
        raise ValueError("Counts must satisfy 0 <= failure_count <= sample_count.")
    return normalized


def restrict_to_cutoff(data: pd.DataFrame, cutoff_month: Any) -> pd.DataFrame:
    """Return only rows available by the inclusive decision-time cutoff."""

    normalized = validate_aggregates(data)
    cutoff = as_month(cutoff_month)
    history = normalized.loc[normalized["month"] <= cutoff].copy()
    if not history.empty and history["month"].max() > cutoff:
        raise AssertionError("Future information entered the historical slice.")
    return history


def aggregate_counts(
    data: pd.DataFrame,
    cutoff_month: Any,
    *,
    window_months: int | None = None,
    start_month: Any | None = None,
    key_columns: Sequence[str] | None = None,
    key_values: Sequence[Any] | None = None,
) -> tuple[float, float]:
    """Aggregate sample and failure counts for an optional key and trailing window."""

    history = restrict_to_cutoff(data, cutoff_month)
    cutoff = as_month(cutoff_month)
    if start_month is not None:
        history = history.loc[history["month"] >= as_month(start_month)]
    if window_months is not None:
        if int(window_months) <= 0:
            raise ValueError("window_months must be positive.")
        start = cutoff - (int(window_months) - 1)
        history = history.loc[history["month"] >= start]
    columns = list(key_columns or [])
    values = list(key_values or [])
    if len(columns) != len(values):
        raise ValueError("key_columns and key_values must have equal length.")
    for column, value in zip(columns, values, strict=True):
        if column not in history.columns:
            raise ValueError(f"Unknown aggregation key column: {column}")
        history = history.loc[history[column] == value]
    return float(history["sample_count"].sum()), float(history["failure_count"].sum())


def empirical_prior(
    data: pd.DataFrame,
    cutoff_month: Any,
    default_prior: float,
    prior_start_month: Any = "2024-01",
) -> float:
    """Estimate the global prior from the inclusive configured prior interval."""

    history = restrict_to_cutoff(data, cutoff_month)
    history = history.loc[history["month"] >= as_month(prior_start_month)]
    if history.empty:
        if not 0.0 <= float(default_prior) <= 1.0:
            raise ValueError("default_prior must lie in [0, 1].")
        return float(default_prior)
    samples, failures = aggregate_counts(history, cutoff_month)
    if samples == 0:
        raise ValueError("The prior interval contains records but no samples; its rate is undefined.")
    return failures / samples


def log_saturated_support(n: float, n_max: float) -> float:
    """Compute Eq. (10), clipped to the unit interval."""

    if not isfinite(n) or not isfinite(n_max) or n < 0 or n_max <= 0:
        raise ValueError("n must be nonnegative and n_max must be positive and finite.")
    return float(min(1.0, max(0.0, log1p(n) / log1p(n_max))))


def beta_rate(failures: float, samples: float, prior_mean: float, prior_strength: float) -> float:
    """Compute the proposed empirical-Bayes rate with equivalent prior sample size."""

    if samples < 0 or failures < 0 or failures > samples:
        raise ValueError("Counts must satisfy 0 <= failures <= samples.")
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive.")
    if not 0.0 <= prior_mean <= 1.0:
        raise ValueError("prior_mean must lie in [0, 1].")
    return float((failures + prior_strength * prior_mean) / (samples + prior_strength))

