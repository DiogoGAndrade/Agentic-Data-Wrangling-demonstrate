import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


def _severity_from_ratio(r: float, warn: float, crit: float) -> str:
    if r >= crit:
        return "critical"
    if r >= warn:
        return "warning"
    return "info"


def check_target_distribution(
    df: pd.DataFrame,
    target_column: Optional[str],
    params: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Prognostic (non-executable): checks target distribution imbalance.

    Returns a Prognostic-like dict or None if not applicable.
    """
    if not target_column:
        return None
    if target_column not in df.columns:
        return {
            "code": "check_target_distribution",
            "severity": "warning",
            "message": f"Target column '{target_column}' not found; cannot compute target distribution.",
            "evidence": {"target_column": target_column},
        }

    s = df[target_column]
    if s.dropna().empty:
        return {
            "code": "check_target_distribution",
            "severity": "warning",
            "message": "Target column contains only missing values; cannot compute distribution.",
            "evidence": {"target_column": target_column},
        }

    # Classification if low unique count; otherwise treat as regression-like
    nunique = int(s.nunique(dropna=True))
    evidence: Dict[str, Any] = {"target_column": target_column, "nunique": nunique}

    if nunique <= 20:
        vc = s.value_counts(dropna=True)
        total = int(vc.sum())
        ratios = (vc / total).to_dict()
        # imbalance ratio = max class share
        max_share = float(max(ratios.values())) if ratios else 0.0
        severity = _severity_from_ratio(max_share, warn=0.70, crit=0.90)

        msg = (
            f"Target appears categorical (nunique={nunique}). "
            f"Max class share is {max_share:.2f}. "
            "Consider class imbalance handling if training a classifier."
        )

        evidence.update(
            {
                "type": "classification_like",
                "class_counts": {str(k): int(v) for k, v in vc.to_dict().items()},
                "class_ratios": {str(k): float(v) for k, v in ratios.items()},
                "max_class_share": max_share,
            }
        )

        return {
            "code": "check_target_distribution",
            "severity": severity,
            "message": msg,
            "evidence": evidence,
        }

    # Regression-like: just show summary stats
    stats = {
        "count": int(s.count()),
        "missing": int(s.isna().sum()),
    }
    if pd.api.types.is_numeric_dtype(s):
        stats.update(
            {
                "mean": float(s.mean()),
                "std": float(s.std(ddof=0)),
                "min": float(s.min()),
                "max": float(s.max()),
            }
        )

    return {
        "code": "check_target_distribution",
        "severity": "info",
        "message": f"Target appears continuous (nunique={nunique}). Review distribution before regression.",
        "evidence": {"target_column": target_column, "type": "regression_like", "stats": stats},
    }


def flag_redundant_or_constant_features(
    df: pd.DataFrame,
    target_column: Optional[str],
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prognostic (non-executable): flags constant / near-constant columns.
    Deterministic, no model fitting.

    params:
      - near_constant_threshold: float (default 0.99)
      - max_report: int (default 20)
    """
    params = params or {}
    thr = float(params.get("near_constant_threshold", 0.99))
    max_report = int(params.get("max_report", 20))

    cols = [c for c in df.columns if c != target_column]
    constant = []
    near_constant = []

    for c in cols:
        s = df[c]
        if s.dropna().empty:
            constant.append(c)
            continue

        vc = s.value_counts(dropna=True)
        top_ratio = float(vc.iloc[0] / vc.sum()) if len(vc) > 0 else 1.0

        if int(s.nunique(dropna=True)) <= 1:
            constant.append(c)
        elif top_ratio >= thr:
            near_constant.append((c, top_ratio))

    severity = "info"
    if constant:
        severity = "warning"
    if len(constant) >= 5:
        severity = "critical"

    msg = (
        f"Found {len(constant)} constant columns and {len(near_constant)} near-constant columns "
        f"(top value ratio >= {thr}). Consider dropping or reviewing them."
    )

    evidence = {
        "target_column": target_column,
        "near_constant_threshold": thr,
        "constant_columns": constant[:max_report],
        "near_constant_columns": [{"column": c, "top_ratio": r} for c, r in near_constant[:max_report]],
        "counts": {"constant": len(constant), "near_constant": len(near_constant)},
    }

    return {
        "code": "flag_redundant_or_constant_features",
        "severity": severity,
        "message": msg,
        "evidence": evidence,
    }


def flag_leakage(
    df: pd.DataFrame,
    target_column: Optional[str],
    params: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Prognostic (non-executable): flags simple leakage signals.

    Deterministic heuristics:
      - any feature identical to target (exact match)
      - any feature highly correlated with numeric target (abs corr >= threshold)
      - any feature name containing target name tokens (weak signal)

    params:
      - corr_threshold: float (default 0.98)
      - max_report: int (default 20)
    """
    if not target_column:
        return None
    if target_column not in df.columns:
        return {
            "code": "flag_leakage",
            "severity": "warning",
            "message": f"Target column '{target_column}' not found; cannot compute leakage checks.",
            "evidence": {"target_column": target_column},
        }

    params = params or {}
    corr_threshold = float(params.get("corr_threshold", 0.98))
    max_report = int(params.get("max_report", 20))

    target = df[target_column]
    candidates = [c for c in df.columns if c != target_column]

    identical = []
    high_corr = []
    name_suspects = []

    # Name suspects (weak)
    t = target_column.lower()
    for c in candidates:
        if t in c.lower():
            name_suspects.append(c)

    # Exact identical check (works for any dtype if aligned)
    for c in candidates:
        s = df[c]
        # Compare where both not NA
        mask = (~s.isna()) & (~target.isna())
        if mask.any():
            if (s[mask].astype("string").values == target[mask].astype("string").values).all():
                identical.append(c)

    # Correlation check only for numeric target and numeric features
    if pd.api.types.is_numeric_dtype(target):
        num_df = df[candidates].select_dtypes(include=["number"])
        if not num_df.empty:
            corr = num_df.corrwith(target)
            for c, v in corr.dropna().items():
                if abs(float(v)) >= corr_threshold:
                    high_corr.append({"column": c, "corr": float(v)})

    severity = "info"
    if identical or high_corr:
        severity = "critical"
    elif name_suspects:
        severity = "warning"

    msg = "No strong leakage signals detected."
    if identical:
        msg = f"Found features identical to target: {identical[:max_report]} (strong leakage)."
    elif high_corr:
        msg = f"Found highly correlated numeric features with target (|corr|>={corr_threshold}). Potential leakage."

    evidence = {
        "target_column": target_column,
        "identical_to_target": identical[:max_report],
        "high_corr_features": high_corr[:max_report],
        "name_suspects": name_suspects[:max_report],
        "corr_threshold": corr_threshold,
    }

    return {
        "code": "flag_leakage",
        "severity": severity,
        "message": msg,
        "evidence": evidence,
    }


def recommend_feature_selection(
    df: pd.DataFrame,
    target_column: Optional[str],
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prognostic recommendation (non-executable): suggests feature selection consideration.

    Deterministic signal:
      - high dimensionality vs rows
      - many constant/near-constant features (reuse simple counts)
    """
    params = params or {}
    warn_ratio = float(params.get("p_over_n_warn", 0.5))    # p/n
    crit_ratio = float(params.get("p_over_n_crit", 1.0))    # p/n

    n_rows = int(len(df))
    p = int(df.shape[1] - (1 if target_column in df.columns else 0))
    ratio = (p / n_rows) if n_rows > 0 else 0.0

    severity = _severity_from_ratio(ratio, warn=warn_ratio, crit=crit_ratio)

    msg = (
        f"Feature selection may be beneficial: p={p} features, n={n_rows} rows (p/n={ratio:.2f}). "
        "Consider dropping redundant/constant features or using regularization."
    )

    return {
        "code": "recommend_feature_selection",
        "severity": severity,
        "message": msg,
        "evidence": {"target_column": target_column, "n_rows": n_rows, "n_features": p, "p_over_n": ratio},
    }


def recommend_oversampling(
    df: pd.DataFrame,
    target_column: Optional[str],
    params: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Prognostic recommendation (non-executable): suggests oversampling if class imbalance seems high.

    Applies only when target seems classification-like (nunique <= 20).
    """
    if not target_column or target_column not in df.columns:
        return None

    s = df[target_column]
    nunique = int(s.nunique(dropna=True))
    if nunique == 0:
        return None
    if nunique > 20:
        return None  # regression-like -> oversampling not relevant

    vc = s.value_counts(dropna=True)
    total = int(vc.sum())
    if total == 0:
        return None

    ratios = (vc / total).to_dict()
    max_share = float(max(ratios.values())) if ratios else 0.0

    # Recommendation thresholds aligned with distribution check
    severity = _severity_from_ratio(max_share, warn=0.70, crit=0.90)

    msg = (
        f"Oversampling may help: target is categorical (nunique={nunique}) with max class share {max_share:.2f}. "
        "Consider SMOTE/RandomOverSampler during model training (not applied in wrangling)."
    )

    evidence = {
        "target_column": target_column,
        "nunique": nunique,
        "class_counts": {str(k): int(v) for k, v in vc.to_dict().items()},
        "class_ratios": {str(k): float(v) for k, v in ratios.items()},
        "max_class_share": max_share,
    }

    return {
        "code": "recommend_oversampling",
        "severity": severity,
        "message": msg,
        "evidence": evidence,
    }


def compute_prognostics(
    df: pd.DataFrame,
    target_column: Optional[str]
) -> List[Dict[str, Any]]:
    """
    Compute all deterministic prognostics for the current dataset.
    Returns list of Prognostic-like dicts (schema-compatible).
    """
    out: List[Dict[str, Any]] = []

    p1 = check_target_distribution(df, target_column)
    if p1:
        out.append(p1)

    p2 = flag_leakage(df, target_column)
    if p2:
        out.append(p2)

    out.append(flag_redundant_or_constant_features(df, target_column))
    out.append(recommend_feature_selection(df, target_column))

    p5 = recommend_oversampling(df, target_column)
    if p5:
        out.append(p5)

    return out