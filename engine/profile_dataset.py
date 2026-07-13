# engine/profile_dataset.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def _safe_examples(series: pd.Series, max_n: int = 5) -> List[Any]:
    vals = series.dropna().astype(str).unique().tolist()
    return vals[:max_n]


def build_dataset_profile(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    max_example_values: int = 5,
) -> Dict[str, Any]:
    """
    Build a compact deterministic profile for LLM planning.

    The goal is to give the planner much more context than head(8)
    without sending the full dataset.
    """
    profile: Dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": [],
        "target_profile": None,
    }

    for col in df.columns:
        s = df[col]

        col_profile: Dict[str, Any] = {
            "name": col,
            "dtype": str(s.dtype),
            "missing_count": int(s.isna().sum()),
            "missing_ratio": float(round(s.isna().mean(), 6)),
            "n_unique": int(s.nunique(dropna=True)),
            "sample_values": _safe_examples(s, max_n=max_example_values),
            "is_numeric": bool(pd.api.types.is_numeric_dtype(s)),
            "is_text_like": bool(pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)),
            "is_constant": bool(s.nunique(dropna=True) <= 1),
        }

        if pd.api.types.is_numeric_dtype(s):
            non_na = s.dropna()
            if len(non_na) > 0:
                col_profile["numeric_summary"] = {
                    "min": float(non_na.min()),
                    "max": float(non_na.max()),
                    "mean": float(round(non_na.mean(), 6)),
                    "median": float(round(non_na.median(), 6)),
                }
            else:
                col_profile["numeric_summary"] = None
        else:
            col_profile["numeric_summary"] = None

        profile["columns"].append(col_profile)

    if target_column and target_column in df.columns:
        target = df[target_column].astype(str)
        counts = target.value_counts(dropna=False).to_dict()
        profile["target_profile"] = {
            "target_column": target_column,
            "class_counts": {str(k): int(v) for k, v in counts.items()},
            "n_unique": int(df[target_column].nunique(dropna=True)),
            "missing_count": int(df[target_column].isna().sum()),
        }

    return profile