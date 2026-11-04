# Risk–Support Decoupling: Core Method Reference Implementation

Selected core methodological components accompanying:

**“Decoupling failure risk from sample support for capacity-constrained food safety testing under selective observation.”**

## Release Scope and Policy

This repository provides selected computational modules for inspecting the core methodological approach.

**The latest version of the code is not publicly available at the request of the data owner.**

- **Included:** Reference modules for risk–support decoupling, feature construction, adaptive hierarchy scoring, monthly training, score smoothing, and capacity-constrained ranking.
- **Excluded:** Proprietary data pipelines, production infrastructure, and the latest internal implementation, including components involving sensitive information or restricted design details.
- **Data availability:** No datasets, raw samples, or pretrained models are included. Users must supply their own data in the required format.
- **Implementation status:** The released files are a **candidate reference implementation**. Their equivalence to the implementation used to produce the paper's reported results has not yet been verified. This release supports inspection of the supplied components and does not establish exact numerical reproduction of those results.

## Included Components

The released source files are in the `risk_support_decoupling/` package.

| Module | Functionality |
| --- | --- |
| `__init__.py` | Package initialization and configuration-loader export |
| `evidence_rates.py` | Monthly aggregate validation, historical cutoff filtering, count aggregation, empirical priors, and empirical Bayes rates |
| `feature_construction.py` | Monthly feature construction and supervised rows with next-month observation masks |
| `hierarchy.py` | Candidate adaptive hierarchy scoring and support provenance |
| `monthly_training.py` | Month-specific gradient boosting and fallback scoring |
| `backward_smoothing.py` | Backward-only score smoothing |
| `capacity_ranking.py` | Configurable ranking, tie-breaking, and fixed-capacity selection |
| `config_loader.py` | YAML configuration loading and partial validation |

The modules use relative imports. Import them through the package rather than executing individual files as standalone scripts.

## Dependencies and Usage

### Prerequisites

- Python 3.10 or later
- NumPy, pandas, scikit-learn, and PyYAML
- A scikit-learn version supporting `OneHotEncoder(sparse_output=False)`

Dependency versions have not been pinned or validated for this release.

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

Counts must satisfy `0 <= failure_count <= sample_count`. Category and hierarchy fields depend on the supplied configuration.

### Configuration

No YAML configuration file is included. Users must supply the parameters referenced by the source, including priors, support parameters, historical windows, hierarchy definitions, feature lists, model settings, smoothing settings, fallback selection, missing-category handling, and tie-breaking rules.

The configuration loader requires this implementation-status marker:

```yaml
method_status: candidate_reference_implementation_pending_author_confirmation
```

This marker is one field in the configuration, not a complete configuration. The checks in `config_loader.py` validate only part of the parameters used downstream. Paper-verified parameter values are not supplied in this release.

## Reproducibility

The public release contains the modules listed above. It does not include experiment configurations, an end-to-end experiment runner, or deployment integrations. Reproducing the paper's numerical results requires the corresponding experimental resources and verification that the implementation and settings match those used for the reported experiments.

The release check covered source inspection, common sensitive-information patterns, and Python syntax. Runtime behavior, numerical results, and equivalence to the paper's experimental implementation have not been validated in that check.
