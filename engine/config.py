"""
Centralised configuration for the wrangling engine.

Single seed governs all stochastic operations across the pipeline:
- 5-fold CV splitter
- IterativeImputer (MICE)
- RandomForest, GradientBoosting
- numpy RNG used in data/scripts/inject_perturbations.py
"""

RANDOM_STATE: int = 369
"""Single random seed for the whole project. Do NOT shadow this constant
with local seeds elsewhere; import it from here."""


TASK_TYPE = {
    # Phase A - model selection (3 classification + 1 regression)
    "adult":            "classification",
    "diabetes":         "classification",
    "student":          "classification",
    "life_expectancy":  "regression",
    # Phase B - held-out generalisation
    "house_prices":     "regression",
    "heart":            "classification",
    "bank":             "classification",
    # Phase C - high-missingness stress test
    # Datasets selected to evaluate C4 under elevated data quality challenges
    # (high natural missing rates, clinical/real-world structure)
    "platform":         "classification",   # CLV: 48.9% missing in age; target=purchased
    "support2_clf":     "classification",   # UCI SUPPORT2: target=hospdead (in-hospital death)
    "support2_reg":     "regression",       # UCI SUPPORT2: target=log_charges (hospitalisation cost)
}
