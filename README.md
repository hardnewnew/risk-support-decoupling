# Risk–Support Decoupling

Selected core code accompanying **“Decoupling failure risk from sample support for capacity-constrained food safety testing under selective observation.”**

## Release scope

This repository makes selected key implementation components available for inspection and research use. Parts of the implementation involving sensitive information or restricted design details are not included in this public release. It provides a reference for the released components rather than the complete research and deployment system.

The supplied source files are marked as a **candidate reference implementation**. Their equivalence to the implementation used to produce the reported paper results has not yet been verified. This release does not establish exact reproduction of the paper's numerical results.

## Included components

| Module | Contents |
| --- | --- |
| `evidence_rates.py` | Monthly aggregate validation, historical cutoff filtering, count aggregation, empirical priors, and empirical Bayes rates |
| `feature_construction.py` | Monthly feature construction and supervised rows with next-month observation masks |
| `hierarchy.py` | Candidate adaptive hierarchy scoring and support provenance |
| `monthly_training.py` | Month-specific gradient boosting and fallback scoring |
| `backward_smoothing.py` | Backward-only score smoothing |
| `capacity_ranking.py` | Configurable ranking, tie-breaking, and fixed-capacity selection |
| `config_loader.py` | YAML configuration loading and partial validation |

All eight supplied Python files, including `__init__.py`, are located in the `risk_support_decoupling/` package. The source uses relative imports, so import the modules as part of this package rather than running individual files as standalone scripts.

## Dependencies and use

The source requires Python 3.10 or later and imports NumPy, pandas, scikit-learn, and PyYAML. Dependency versions have not been pinned or validated for this release. The scikit-learn installation must support the `OneHotEncoder(sparse_output=False)` argument used by the code.

With the repository root on the Python import path and the dependencies installed, the modules can be imported as follows:

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

These imports identify the available interfaces; they are not an end-to-end reproduction script. Function inputs and outputs are documented in the source.

## Inputs and configuration

The released functions operate on user-supplied pandas DataFrames. Monthly aggregates require `month`, `risk_unit_id`, `sample_count`, and `failure_count`. Category and hierarchy fields depend on the supplied configuration. The code requires counts to satisfy `0 <= failure_count <= sample_count`.

No configuration file is included. Users must supply the parameters referenced by the modules, including prior and support parameters, historical windows, hierarchy definitions, feature lists, model settings, smoothing settings, fallback selection, missing-category handling, and tie-breaking rules. The checks in `config_loader.py` cover only part of the configuration used downstream.

The configuration loader also requires the following implementation-status field:

```yaml
method_status: candidate_reference_implementation_pending_author_confirmation
```

This field is a status marker, not a complete configuration. The release supplies no paper-verified parameter values.

## Availability and reproducibility

The supplied package contains code only. It does not include datasets, trained models, experiment configurations, an end-to-end experiment runner, or deployment integrations. No withheld data or restricted implementation details are embedded in this repository's release documentation.

Running the released components requires compatible dependencies, suitable input data, and a complete configuration. Reproducing all reported experiments additionally requires the corresponding experimental resources and verification that the implementation and settings match those used for the paper.

The pre-publication check covered source inspection, common sensitive-information patterns, and Python syntax. Runtime behavior, numerical results, and equivalence to the paper's experimental implementation have not been validated in that check.
