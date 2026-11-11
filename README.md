# An Evidence-aware Decision-support Framework for Capacity-constrained Food Safety Testing under Selective Observation

Selected core methodological components accompanying:

**“An Evidence-aware Decision-support Framework for Capacity-constrained Food Safety Testing under Selective Observation.”**

## Release Scope and Policy

This repository provides selected computational modules for inspecting the core methodological approach.

**The latest version of the code is not publicly available at the request of the data owner.**

- **Included:** Reference modules for risk–support decoupling, feature construction, adaptive hierarchy scoring, monthly training, score smoothing, and capacity-constrained ranking.
- **Excluded:** Proprietary data pipelines, production infrastructure, and the latest internal implementation, including components involving sensitive information or restricted design details.
- **Data availability:** No datasets, raw samples, or pretrained models are included. Users must supply their own data in the required format.
- **Implementation status:** The released modules are a reference implementation written to follow the method specification in the paper (Eqs. (9)–(11), Table 2, Algorithm 1, Appendices D.1, E.1 and E.2, and Table E.1). Because the data and the latest internal implementation cannot be released, this code has not been run on the paper's data and does not by itself reproduce the reported numbers.

## Included Components

The released source files are in the `risk_support_decoupling/` package.

| Module | Functionality |
| --- | --- |
| `__init__.py` | Package initialization and configuration-loader export |
| `evidence_rates.py` | Monthly aggregate validation, historical cutoff filtering, count aggregation, empirical priors, and empirical Bayes rates |
| `feature_construction.py` | Monthly feature construction and supervised rows with next-month observation masks |
| `hierarchy.py` | Ordered Appendix E.1 backoff scoring with window, count, and prior provenance |
| `monthly_training.py` | Monthly System LightGBM or GBM-9 scoring and configured cold-start fallback |
| `backward_smoothing.py` | Optional backward-only score smoothing (disabled by default) |
| `capacity_ranking.py` | Configurable ranking, tie-breaking, and fixed-capacity selection |
| `config_loader.py` | YAML configuration loading and partial validation |

The modules use relative imports. Import them through the package rather than executing individual files as standalone scripts.

## Dependencies and Usage

### Prerequisites

- Python 3.10 or later
- NumPy, pandas, scikit-learn >= 1.2, PyYAML, and LightGBM
- pytest for the synthetic tests

Dependencies are listed in `requirements.txt` without exact version pins.

### Importing the Modules

With the dependencies installed and the repository root on the Python import path:

```python
from risk_support_decoupling.config_loader import load_reference_config
from risk_support_decoupling.feature_construction import (
    construct_features_for_month,
    build_supervised_rows,
)
from risk_support_decoupling.monthly_training import fit_or_fallback
from risk_support_decoupling.backward_smoothing import smooth_scores
from risk_support_decoupling.capacity_ranking import rank_and_select
```

This example shows the available imports. Running the components requires input data and a complete configuration. Function signatures and docstrings are provided in each module.

## Inputs and Configuration

### Data Schema

The functions operate on pandas DataFrames. Monthly aggregates require:

- `month`
- `risk_unit_id`
- `sample_count`
- `failure_count`
- `broad_category`
- `category`
- `sampling_stage`
- `tested_item`

Counts must satisfy `0 <= failure_count <= sample_count`. The string `risk_unit_id` identifies the four-field tuple `(broad_category, category, sampling_stage, tested_item)`. Each key must map one-to-one to a tuple; enterprise identifiers are excluded from unit definitions and model inputs. The default model uses the seven numeric fields in Table 2, followed by `broad_category` and `sampling_stage`. The other unit-key fields and audit columns are not model inputs.

### Configuration

`configs/paper_reference.yaml` supplies the specified paper parameters and explicitly documented implementation choices. Load it with:

```python
config = load_reference_config("configs/paper_reference.yaml")
```

The default System uses LightGBM with the Table E.1 settings. The `gbm9_sklearn` option retains the GBM-9 comparator. Optional `learner_params` overrides are supported but are absent from the included configuration; using them departs from the specified defaults. Comments in the configuration describe the GBM-11 ablation, including its additional fields and fallback settings.

For decision month `m` and configured delay `L`, features and the global prior use records no later than `m - L`. Unit counts also respect `local_history_start_month`, while the prior independently respects `prior_start_month`. Each training row for decision `d` rebuilds its own features and prior at `d - L`, with the label from `d + 1`. At training cutoff `m`, rows must satisfy `d + 1 <= m - L`. Unobserved labels remain missing and are excluded from training.

The implementation-status marker is:

```yaml
method_status: reference_implementation_following_paper_specification
```

Call the modules in the order appropriate to the task: construct historical supervised rows and current features, fit or apply the fallback, call `smooth_scores`, then `rank_and_select` with an integer capacity. Calling `smooth_scores` supplies the `smoothed_score` column even when smoothing is disabled, as expected by the included ranking configuration.

## Paper-to-code Map

| Paper element | Code |
| --- | --- |
| Eq. (9) | `evidence_rates.beta_rate` |
| Eq. (10) | `evidence_rates.log_saturated_support` |
| Table 2 | `feature_construction.construct_features_for_month` |
| Algorithm 1, steps 2–3 | `feature_construction.build_supervised_rows` and `monthly_training.fit_or_fallback` |
| Eq. (11) | `backward_smoothing.smooth_scores` |
| Algorithm 1, step 5 | `capacity_ranking.rank_and_select` |
| Appendix E.1 hierarchy backoff | `hierarchy.adaptive_hierarchy_score` |
| Table E.1 System | `monthly_training.fit_or_fallback`, with `learner: system_lightgbm` |

## Implementation Choices Not Fixed by the Paper

- `month_of_year` uses the decision-cutoff month, including when reporting latency moves the historical cutoff to an earlier month.
- Group backoff uses the entire local history by default (`group_backoff_window_months: null`). A positive month count restricts this window while still respecting the local start.
- Both learners retain one-hot categorical encoding with unknown categories ignored. Preprocessing is fitted exclusively on eligible, observed training rows at the current cutoff.
- Numeric missing values are filled with zero, with no scaling. Missing category values use the existing configurable token, `__missing__` in the included configuration.
- `random_state: 0` is an implementation seed, not a paper-specified value.
- Capacity is passed as a nonnegative integer. Rounding a proportional budget into an integer capacity is left to the caller.
- Score smoothing is optional and disabled by default. When enabled, its default coefficient is 0.6; first appearances retain their current score. The existing update uses the latest available earlier score per unit, including across missing months.
- The default candidate pool consists of units observed by the effective historical cutoff. The existing explicit candidate-table interface is retained; callers are responsible for providing attributes available at that cutoff.
- `default_prior: 0.0` applies only when the prior interval contains no records. If records exist but total samples are zero, the requested prior formula is undefined; the implementation raises an error rather than substitutes an unspecified value.
- The input string key is retained. Keys are checked against the four-field unit tuple rather than rebuilt using a newly introduced delimiter. Monthly duplicate aggregates are summed, as in the existing implementation.
- Provenance retains zero-based source indexes and the inclusive trailing-window convention. Rejected model inputs are enforced through an allowlist of Table 2 fields and the two specified ablation fields, preventing identifiers from entering either learner.

## Not Included

- The offline feature-governance engine and its 2 × 2 crossed design
- Log hit-ratio decomposition
- Manski bounds
- Capture evaluation
- The Chicago and Hangzhou processing and experiment pipelines
- Sample-level provenance archiving
- Production deployment integrations

## Reproducibility and Checks

The released modules and `configs/paper_reference.yaml` follow the method specification in the paper. The tests construct synthetic values in memory and read no external data files. No datasets or standalone data examples are distributed. Without the paper's data and the latest internal implementation, these modules do not by themselves reproduce the reported numerical results.

From the repository root, run:

```sh
python -m pytest -q
python -m compileall risk_support_decoupling
```

With the packages in `requirements.txt` installed, all synthetic tests in `tests/` pass. Synthetic checks address implementation behavior; they do not establish reproduction of the paper's empirical findings.
