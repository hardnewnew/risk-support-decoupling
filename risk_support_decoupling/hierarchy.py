"""Candidate reference implementation; not yet verified as identical to the implementation used for the reported results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .evidence_rates import aggregate_counts, as_month, beta_rate, empirical_prior, restrict_to_cutoff


@dataclass(frozen=True)
class HierarchyScore:
    """Candidate hierarchy result with auditable support provenance."""

    score: float
    selected_level: str
    selected_level_index: int | None
    selected_window_months: int | None
    support_count: float
    prior: float


def _normalized_value(value: Any, missing_token: str) -> str:
    """Map missing hierarchy/category values to a deterministic token."""

    return missing_token if pd.isna(value) else str(value)


def _matching_history(
    history: pd.DataFrame,
    candidate: dict[str, Any],
    level_columns: list[str],
    missing_token: str,
) -> pd.DataFrame:
    """Select rows sharing the candidate's configured hierarchy key."""

    selected = history
    for column in level_columns:
        if column not in selected.columns:
            selected = selected.assign(**{column: missing_token})
        row_values = selected[column].map(lambda value: _normalized_value(value, missing_token))
        candidate_value = _normalized_value(candidate.get(column), missing_token)
        selected = selected.loc[row_values == candidate_value]
    return selected


def _counts(data: pd.DataFrame, cutoff_month: Any, window_months: int | None) -> tuple[float, float]:
    """Aggregate counts from an already key-filtered history."""

    return aggregate_counts(data, cutoff_month, window_months=window_months)


def adaptive_hierarchy_score(
    data: pd.DataFrame,
    candidate: dict[str, Any],
    cutoff_month: Any,
    config: dict[str, Any],
) -> HierarchyScore:
    """Apply the proposed shortest-window hierarchy/backoff and parent shrinkage rule."""

    history = restrict_to_cutoff(data, cutoff_month)
    cutoff = as_month(cutoff_month)
    prior = empirical_prior(history, cutoff, float(config["default_prior"]))
    levels = [list(level) for level in config["hierarchy_levels"]]
    windows = sorted({int(value) for value in config["adaptive_windows_months"]})
    kappa = float(config["kappa"])
    tau = float(config["tau"])
    m0 = float(config["m0"])
    missing_token = str(config["missing_category_token"])
    if kappa < 0 or tau <= 0:
        raise ValueError("kappa must be nonnegative and tau must be positive.")

    for level_index, level_columns in enumerate(levels):
        level_history = _matching_history(history, candidate, level_columns, missing_token)
        selected_window: int | None = None
        selected_samples = 0.0
        selected_failures = 0.0
        for window in windows:
            samples, failures = _counts(level_history, cutoff, window)
            if samples >= kappa:
                selected_window = window
                selected_samples, selected_failures = samples, failures
                break
        if selected_window is None:
            samples, failures = _counts(level_history, cutoff, None)
            if samples < kappa:
                continue
            selected_samples, selected_failures = samples, failures

        selected_rate = beta_rate(selected_failures, selected_samples, prior, m0)
        if level_index + 1 < len(levels):
            parent_columns = levels[level_index + 1]
            parent_history = _matching_history(history, candidate, parent_columns, missing_token)
            parent_samples, parent_failures = _counts(parent_history, cutoff, selected_window)
            parent_rate = beta_rate(parent_failures, parent_samples, prior, m0)
        else:
            parent_rate = prior
        weight = selected_samples / (selected_samples + tau) if selected_samples > 0 else 0.0
        score = weight * selected_rate + (1.0 - weight) * parent_rate
        level_name = "+".join(level_columns) if level_columns else "global"
        return HierarchyScore(
            score=float(score),
            selected_level=level_name,
            selected_level_index=level_index,
            selected_window_months=selected_window,
            support_count=selected_samples,
            prior=prior,
        )

    return HierarchyScore(
        score=prior,
        selected_level="global_prior",
        selected_level_index=None,
        selected_window_months=None,
        support_count=0.0,
        prior=prior,
    )

