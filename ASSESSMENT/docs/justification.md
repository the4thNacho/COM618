# Library and Tool Justification

## Overview

This document explains each library chosen for the COM618 assessment pipeline and
justifies the decisions made in terms of suitability, industry adoption, and
alternatives considered.

---

## Core Libraries

### pandas
**Version**: ≥ 2.0.0  
**Used for**: Loading, cleaning, and transforming CSV data throughout the pipeline.

**Justification**: pandas is the de-facto standard for tabular data manipulation in
Python. It provides the `DataFrame` abstraction that makes column-wise operations
(imputation, value-counts, groupby) concise and readable. The alternative `polars`
is faster for large datasets but has a less mature API and fewer tutorials —
inappropriate for an educational context. `csv` from the standard library would
require writing all aggregation and imputation logic from scratch.

---

### NumPy
**Version**: ≥ 1.24.0  
**Used for**: Numerical operations (median, mean, array construction for model input).

**Justification**: NumPy underpins pandas and scikit-learn; it cannot meaningfully
be avoided. Its vectorised array operations are orders of magnitude faster than
Python loops for numerical work.

---

### Matplotlib
**Version**: ≥ 3.7.0  
**Used for**: All baseline charts (histograms, bar charts, box plots, scatter plots).

**Justification**: Matplotlib is the foundational plotting library in Python with
the widest compatibility across backends (including the `Agg` non-interactive
backend used by the Flask server). It gives full control over every visual element.

---

### Seaborn
**Version**: 0.13.2  
**Used for**: Statistical charts — heatmaps, enhanced box plots, grouped visualisations.

**Justification**: Seaborn is built on top of Matplotlib and provides high-level
functions for statistical graphics that would require many lines of raw Matplotlib
code. The `heatmap` function for correlation matrices and `boxplot` with categorical
grouping are particularly useful here. Plotly was considered as an alternative for
interactive charts, but adding a JavaScript dependency to a server-rendered Flask
app was considered unnecessary complexity for this assessment.

---

### scikit-learn
**Version**: 1.8.0  
**Used for**: `RandomForestClassifier`, `train_test_split`, `cross_val_score`,
`LabelEncoder`, `classification_report`, `confusion_matrix`.

**Justification**: scikit-learn is the most widely used general-purpose machine
learning library in Python. Its consistent `fit`/`predict` API, comprehensive
documentation, and built-in evaluation utilities make it the natural choice for
a teaching or assessment context. Alternatives such as XGBoost or LightGBM would
offer marginal performance improvements on this dataset but at the cost of an
additional dependency with less educational familiarity.

---

### Flask
**Version**: latest (≥ 3.0)  
**Used for**: Web server, routing, template rendering.

**Justification**: Flask is a lightweight WSGI micro-framework that imposes minimal
structure, making it appropriate for a data-science project where the application
logic (cleaning, exploration, prediction) is the focus, not the web framework.
Django would be excessive — it includes an ORM, admin panel, and authentication
system that are not needed here. FastAPI would be a viable modern alternative but
introduces async complexity; Flask's synchronous model is simpler to reason about
for a single-user academic tool. Jinja2 (Flask's built-in template engine) allows
clean separation between Python business logic and HTML presentation.

---

## Serialisation

### pickle (standard library)
**Used for**: Saving and loading the trained Random Forest model and LabelEncoders.

**Justification**: `pickle` is built into Python and is the simplest way to
persist a scikit-learn model. `joblib` (also common for sklearn models) is
marginally faster for models with large NumPy arrays, but the performance
difference is negligible for a 100-tree Random Forest on this data size.
The saved model files are not transferred between machines, so pickle's
security caveats (do not unpickle untrusted data) are not a concern here.

---

## Testing

### unittest (standard library)
**Used for**: Automated unit tests in `tests/`.

**Justification**: `unittest` is built into Python and requires no additional
installation. It provides the `TestCase` class with `setUp`/`setUpClass`
lifecycle hooks, which are useful for expensive setup (training the model once
per test suite). `pytest` is used as the test runner because it provides richer
output and automatic test discovery without requiring explicit test registration.

---

## Decisions on What Was Excluded

| Considered | Decision | Reason |
|---|---|---|
| Tableau | Not used | Requires proprietary licence; matplotlib/seaborn provide equivalent static charts |
| Plotly / Dash | Not used | Interactive charts add JS complexity; static PNGs served by Flask are sufficient |
| XGBoost | Not used | Marginal gain on 52 samples; sklearn RF is simpler and sufficient |
| SQLite / PostgreSQL | Not used | Dataset is a single flat CSV; no relational queries required |
| Docker | Not used | Single-machine development; virtualenv is adequate for dependency isolation |

---

## Environment

All dependencies are installed in a Python virtual environment (`scripts/venv/`).
The full list of direct dependencies is:

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn==0.13.2
scikit-learn==1.8.0
flask>=3.0
pytest
```
