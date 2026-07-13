import re
import pandas as pd
from typing import Dict, Any, Tuple, List
import unicodedata
from sklearn.impute import KNNImputer, SimpleImputer
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # required before the IterativeImputer import below
from sklearn.impute import IterativeImputer, SimpleImputer

def fix_column_names(
    df: pd.DataFrame,
    params: dict | None = None
):
    df_after = df.copy()

    old_cols = list(df.columns)
    new_cols = [
        c.strip().lower().replace(" ", "_").replace("-", "_")
        for c in old_cols
    ]

    df_after.columns = new_cols

    diff = {
        "columns_renamed": dict(zip(old_cols, new_cols))
    }

    warnings = []

    return df_after, diff, warnings

def cast_type(
    df: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """
    Deterministically cast column types.

    Returns:
        df_after
        diff_summary (dict)
        warnings (list of strings)
    """

    warnings = []
    df_after = df.copy()

    columns = params.get("columns")
    dtype = params.get("dtype")
    errors = params.get("errors", "raise")

    if errors not in {"raise", "coerce", "ignore"}:
        raise ValueError("cast_type 'errors' must be one of: raise|coerce|ignore")

    datetime_format = params.get("datetime_format", None)

    if not columns or not isinstance(columns, list):
        raise ValueError("cast_type requires 'columns' as a non-empty list")

    if dtype not in {"int", "float", "bool", "string", "datetime"}:
        raise ValueError(f"Unsupported dtype '{dtype}' in cast_type")

    diff = {
        "columns_casted": [],
        "invalid_values_coerced": {},
        "before_dtypes": {},
        "after_dtypes": {},
    }

    for col in columns:
        if col not in df_after.columns:
            raise ValueError(f"Column '{col}' does not exist")

        diff["before_dtypes"][col] = str(df_after[col].dtype)

        series_before = df_after[col]

        try:
            if dtype == "datetime":
                series_after = pd.to_datetime(
                    series_before,
                    format=datetime_format,
                    errors=errors
                )

            elif dtype in {"int", "float"}:
                series_after = pd.to_numeric(
                    series_before,
                    errors=errors
                )
                if dtype == "int":
                    series_after = series_after.astype("Int64")

            elif dtype == "bool":
                series_after = series_before.astype("boolean")

            elif dtype == "string":
                series_after = series_before.astype("string")

        except Exception as e:
            raise RuntimeError(f"cast_type failed for column '{col}': {e}")

        if errors == "coerce":
            n_coerced = series_after.isna().sum() - series_before.isna().sum()
            if n_coerced > 0:
                diff["invalid_values_coerced"][col] = int(n_coerced)
                warnings.append(
                    f"{n_coerced} invalid values coerced to NaN in column '{col}'"
                )

        df_after[col] = series_after
        diff["after_dtypes"][col] = str(series_after.dtype)
        diff["columns_casted"].append(col)

    return df_after, diff, warnings



def handle_missing(df, params):
    df_after = df.copy()
    diff = {"imputation_values": {}}
    warnings = []
    
    columns = params.get("columns", [])
    strategy = params.get("strategy", "impute")
    
    if not columns:
        return df_after, diff, warnings
        
    # Row-dropping strategy (this caused the C1_manual bug previously)
    if strategy == "drop_rows":
        before_len = len(df_after)
        df_after.dropna(subset=columns, inplace=True)
        diff["dropped_rows"] = before_len - len(df_after)
        return df_after, diff, warnings
        
    # Imputation strategy (MICE-based, our primary approach)
    elif strategy == "impute":
        impute_cfg = params.get("impute", {})
        categorical_strategy = impute_cfg.get("categorical", "most_frequent")
        constant_value = impute_cfg.get("constant_value", 0)
        constant_categorical = impute_cfg.get("constant_categorical", "MISSING")

        num_cols = []
        cat_cols = []
        for c in columns:
            if c not in df_after.columns or df_after[c].isna().sum() == 0:
                continue
            if pd.api.types.is_numeric_dtype(df_after[c]) and not pd.api.types.is_bool_dtype(df_after[c]):
                num_cols.append(c)
            else:
                cat_cols.append(c)

        # --- MICE imputation (IterativeImputer) ---
        if num_cols:
            for c in num_cols:
                s_numeric = pd.to_numeric(df_after[c], errors="coerce")
                n_na_before = int(df_after[c].isna().sum())
                n_na_after = int(s_numeric.isna().sum())
                n_newly_coerced = max(0, n_na_after - n_na_before)

                if n_newly_coerced > 0:
                    warnings.append(f"Column '{c}': {n_newly_coerced} coerced to NaN.")
                    diff.setdefault("coerced_to_nan_before_impute", {})
                    diff["coerced_to_nan_before_impute"][c] = n_newly_coerced
                
                # Fill an all-empty column with a constant fallback
                if s_numeric.dropna().empty:
                    df_after[c] = s_numeric.fillna(constant_value)
                    diff["imputation_values"][c] = constant_value
                else:
                    df_after[c] = s_numeric

            cols_for_mice = [c for c in num_cols if df_after[c].isna().sum() > 0]
            if cols_for_mice:
                # O poderoso MICE
                imputer = IterativeImputer(max_iter=10, random_state=369)
                df_after[num_cols] = imputer.fit_transform(df_after[num_cols])
                for c in cols_for_mice:
                    diff["imputation_values"][c] = "MICE_Estimated"

        # --- Categorical imputation ---
        for c in cat_cols:
            s = df_after[c]
            if s.dropna().empty:
                df_after[c] = s.fillna(constant_categorical)
                diff["imputation_values"][c] = constant_categorical
                continue
            
            if categorical_strategy == "most_frequent":
                modes = s.dropna().astype("string").mode()
                fill_val = str(sorted(modes.tolist())[0]) if len(modes) > 0 else constant_categorical
            else:
                fill_val = constant_categorical
            
            df_after[c] = s.fillna(fill_val)
            diff["imputation_values"][c] = fill_val

    # Mandatory return here (outside the branches above) to avoid crashing
    return df_after, diff, warnings


def _remove_accents(s: str) -> str:
    # Deterministic unicode normalization
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def normalize_text(
    df: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """
    Deterministically normalize text columns.

    params:
      - columns: list[str] (required)
      - ops: list[str] in {"strip","lower","remove_accents","collapse_whitespace"}
      - replace: list[{"pattern": str, "repl": str}] (optional) - regex replacements in order
      - null_like_to_nan: list[str] (optional) - values (case-insensitive after ops) converted to NaN
      - non_string_policy: "coerce_to_string" | "skip" (default "coerce_to_string")
    """

    warnings: List[str] = []
    df_after = df.copy()

    columns = params.get("columns")
    if not columns or not isinstance(columns, list):
        raise ValueError("normalize_text requires 'columns' as a non-empty list")

    ops = params.get("ops", ["strip", "lower", "collapse_whitespace"])
    if not isinstance(ops, list):
        raise ValueError("normalize_text 'ops' must be a list")

    allowed_ops = {"strip", "lower", "remove_accents", "collapse_whitespace"}
    for op in ops:
        if op not in allowed_ops:
            raise ValueError(f"normalize_text op '{op}' is not supported")

    replace_rules = params.get("replace", [])
    if replace_rules and not isinstance(replace_rules, list):
        raise ValueError("normalize_text 'replace' must be a list of {pattern,repl}")

    null_like = params.get("null_like_to_nan", [])
    if null_like and not isinstance(null_like, list):
        raise ValueError("normalize_text 'null_like_to_nan' must be a list of strings")

    non_string_policy = params.get("non_string_policy", "coerce_to_string")
    if non_string_policy not in {"coerce_to_string", "skip"}:
        raise ValueError("normalize_text 'non_string_policy' must be 'coerce_to_string' or 'skip'")

    for col in columns:
        if col not in df_after.columns:
            raise ValueError(f"Column '{col}' does not exist")

    diff: Dict[str, Any] = {
        "columns_normalized": [],
        "values_changed_count": {},
        "null_like_converted_to_nan": {},
        "replacements_applied": {},
    }

    for col in columns:
        s = df_after[col]

        # Work on a copy; preserve NaNs
        s_before = s.copy()

        # Prepare base as string series if needed
        if non_string_policy == "coerce_to_string":
            # Keep NaNs as NaN, not "nan"
            base = s.astype("string")
        else:
            # skip non-string: normalize only where value is str
            base = s.copy()

        def apply_ops(x: Any) -> Any:
            if pd.isna(x):
                return x
            if non_string_policy == "skip" and not isinstance(x, str):
                return x

            # Ensure string
            text = x if isinstance(x, str) else str(x)

            if "strip" in ops:
                text = text.strip()
            if "lower" in ops:
                text = text.lower()
            if "remove_accents" in ops:
                text = _remove_accents(text)
            if "collapse_whitespace" in ops:
                text = re.sub(r"\s+", " ", text).strip()

            # Apply regex replacements in order
            if replace_rules:
                applied = 0
                for rule in replace_rules:
                    pattern = rule.get("pattern")
                    repl = rule.get("repl", "")
                    if not isinstance(pattern, str):
                        raise ValueError("normalize_text replace rule missing 'pattern' string")
                    new_text = re.sub(pattern, repl, text)
                    if new_text != text:
                        applied += 1
                    text = new_text
                if applied > 0:
                    diff["replacements_applied"].setdefault(col, 0)
                    diff["replacements_applied"][col] += applied

            return text

        s_after = base.map(apply_ops)

        # Convert null-like to NaN (case-insensitive match)
        if null_like:
            nl_set = set(str(x).lower() for x in null_like)
            # Only for non-NA values
            mask = s_after.notna() & s_after.astype("string").str.lower().isin(nl_set)
            n_null_like = int(mask.sum())
            if n_null_like > 0:
                s_after = s_after.mask(mask, other=pd.NA)
                diff["null_like_converted_to_nan"][col] = n_null_like

        # Count changes (excluding NaN==NaN)
        # Compare stringified safely
        before_str = s_before.astype("string")
        after_str = s_after.astype("string")
        changed_mask = (before_str != after_str) & ~(before_str.isna() & after_str.isna())
        n_changed = int(changed_mask.sum())

        diff["values_changed_count"][col] = n_changed
        diff["columns_normalized"].append(col)

        if n_changed == 0:
            warnings.append(f"normalize_text: no values changed in column '{col}'")

        df_after[col] = s_after

    return df_after, diff, warnings


def deduplicate(
    df: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """
    Deterministically remove duplicate rows.

    params:
      - subset: list[str] | None  (columns to consider; if None -> full row)
      - keep: "first" | "last" | "none"  (default "first")
      - case_insensitive: bool (default False; only applies to string columns in subset)
    """

    warnings: List[str] = []
    df_after = df.copy()

    subset = params.get("subset", None)
    keep = params.get("keep", "first")
    case_insensitive = params.get("case_insensitive", False)

    if keep not in {"first", "last", "none"}:
        raise ValueError("deduplicate 'keep' must be one of: first | last | none")

    if subset is not None:
        if not isinstance(subset, list) or len(subset) == 0:
            raise ValueError("deduplicate 'subset' must be a non-empty list or None")
        for col in subset:
            if col not in df_after.columns:
                raise ValueError(f"Column '{col}' does not exist")

    n_rows_before = len(df_after)

    # Prepare a comparison frame if case_insensitive is enabled
    if case_insensitive and subset is not None:
        compare_df = df_after.copy()
        for col in subset:
            if pd.api.types.is_string_dtype(compare_df[col]):
                compare_df[col] = compare_df[col].astype("string").str.lower()
    else:
        compare_df = df_after

    # Deduplication logic
    if keep == "none":
        duplicated_mask = compare_df.duplicated(subset=subset, keep=False)
        df_after = df_after.loc[~duplicated_mask]
        n_removed = int(duplicated_mask.sum())
    else:
        duplicated_mask = compare_df.duplicated(subset=subset, keep=keep)
        df_after = df_after.loc[~duplicated_mask]
        n_removed = int(duplicated_mask.sum())

    n_rows_after = len(df_after)

    if n_removed == 0:
        warnings.append("No duplicate rows were found for the given configuration.")

    diff: Dict[str, Any] = {
        "subset": subset,
        "keep": keep,
        "case_insensitive": case_insensitive,
        "rows_before": n_rows_before,
        "rows_after": n_rows_after,
        "rows_removed": n_removed,
    }

    # Optional: show a small sample of duplicate keys (audit-friendly)
    if subset is not None and n_removed > 0:
        dup_examples = (
            compare_df.loc[compare_df.duplicated(subset=subset, keep=False), subset]
            .head(3)
            .to_dict(orient="records")
        )
        diff["duplicate_key_examples"] = dup_examples

    return df_after, diff, warnings


def drop_column(
    df: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """
    Deterministically drop columns.

    params:
      - columns: list[str] (required)
      - reason: str (optional) e.g. "high_missingness|id_like|leakage_suspect|user_request"
      - target_column: str | None (optional)  -> safety: block dropping target unless allow_drop_target=true
      - allow_drop_target: bool (optional, default False)
      - on_missing: "raise" | "skip" (optional, default "raise")  -> behavior when a column is not found
    """

    warnings: List[str] = []
    df_after = df.copy()

    columns = params.get("columns")
    if not columns or not isinstance(columns, list):
        raise ValueError("drop_column requires 'columns' as a non-empty list")

    reason = params.get("reason", None)
    target_column = params.get("target_column", None)
    allow_drop_target = bool(params.get("allow_drop_target", False))
    on_missing = params.get("on_missing", "raise")

    if on_missing not in {"raise", "skip"}:
        raise ValueError("drop_column 'on_missing' must be 'raise' or 'skip'")

    # Safety: block dropping target unless explicitly allowed
    if target_column and (target_column in columns) and not allow_drop_target:
        raise ValueError(
            f"Refusing to drop target column '{target_column}'. "
            "Set params.allow_drop_target=true to override explicitly."
        )

    existing = []
    missing = []
    for c in columns:
        if c in df_after.columns:
            existing.append(c)
        else:
            missing.append(c)

    if missing:
        msg = f"drop_column: columns not found: {missing}"
        if on_missing == "raise":
            raise ValueError(msg)
        warnings.append(msg)

    if not existing:
        warnings.append("drop_column: no columns were dropped (none found).")
        diff = {
            "reason": reason,
            "requested": columns,
            "dropped": [],
            "missing": missing,
            "cols_before": int(df.shape[1]),
            "cols_after": int(df.shape[1]),
        }
        return df_after, diff, warnings

    cols_before = int(df_after.shape[1])
    df_after = df_after.drop(columns=existing)
    cols_after = int(df_after.shape[1])

    diff = {
        "reason": reason,
        "requested": columns,
        "dropped": existing,
        "missing": missing,
        "cols_before": cols_before,
        "cols_after": cols_after,
    }

    # Audit-friendly warning if something critical-looking got dropped
    if target_column and allow_drop_target and target_column in existing:
        warnings.append(f"Target column '{target_column}' was dropped (allow_drop_target=true).")

    return df_after, diff, warnings


def encode_categorical(
    df: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """
    Deterministically encode categorical columns.

    params:
      - columns: list[str] (required)
      - method: "one_hot" | "ordinal" (required)
      - drop_first: bool (optional, default False)  [one_hot]
      - handle_unknown: "ignore" | "error" (optional, default "ignore")
      - max_categories: int (optional, default 30)  [one_hot]
      - other_label: str (optional, default "OTHER") [one_hot, when max_categories exceeded]
      - ordinal_order: "alphabetical" | "frequency" (optional, default "alphabetical") [ordinal]
    Notes:
      - Stores mappings / categories in diff_summary for reproducibility.
      - Does not implement train/test separation here (handled later at pipeline/export stage).
    """

    warnings: List[str] = []
    df_after = df.copy()

    columns = params.get("columns")
    method = params.get("method")

    if not columns or not isinstance(columns, list):
        raise ValueError("encode_categorical requires 'columns' as a non-empty list")

    if method not in {"one_hot", "ordinal"}:
        raise ValueError("encode_categorical requires 'method' in {one_hot, ordinal}")

    for c in columns:
        if c not in df_after.columns:
            raise ValueError(f"Column '{c}' does not exist")

    drop_first = bool(params.get("drop_first", False))
    handle_unknown = params.get("handle_unknown", "ignore")
    if handle_unknown not in {"ignore", "error"}:
        raise ValueError("encode_categorical 'handle_unknown' must be 'ignore' or 'error'")

    max_categories = int(params.get("max_categories", 30))
    if max_categories < 1:
        raise ValueError("encode_categorical 'max_categories' must be >= 1")

    other_label = params.get("other_label", "OTHER")
    ordinal_order = params.get("ordinal_order", "alphabetical")
    if ordinal_order not in {"alphabetical", "frequency"}:
        raise ValueError("encode_categorical 'ordinal_order' must be 'alphabetical' or 'frequency'")

    diff: Dict[str, Any] = {
        "method": method,
        "encoded_columns": [],
        "dropped_original_columns": [],
        "created_columns": [],
        "mappings": {},          # for ordinal: {col: {category: code}}
        "one_hot_categories": {},# for one_hot: {col: [categories_kept] + maybe OTHER}
        "unknown_handling": handle_unknown,
        "drop_first": drop_first,
        "max_categories": max_categories,
        "other_label": other_label,
    }

    if method == "one_hot":
        for col in columns:
            # Work with strings; keep NaNs
            s = df_after[col].astype("string")

            # Determine categories (excluding NA)
            value_counts = s.dropna().value_counts()

            if len(value_counts) == 0:
                warnings.append(f"encode_categorical(one_hot): column '{col}' has only NA values; skipped.")
                continue

            # If too many categories, keep top-K and map rest to OTHER (deterministic)
            if len(value_counts) > max_categories:
                kept = list(value_counts.head(max_categories).index.astype(str))
                warnings.append(
                    f"encode_categorical(one_hot): column '{col}' has {len(value_counts)} categories; "
                    f"keeping top {max_categories} and mapping the rest to '{other_label}'."
                )

                def map_other(x):
                    if pd.isna(x):
                        return x
                    x = str(x)
                    return x if x in kept else other_label

                s_mapped = s.map(map_other)
                categories = sorted(set(kept + [other_label]))
            else:
                s_mapped = s
                categories = sorted(set(value_counts.index.astype(str).tolist()))

            diff["one_hot_categories"][col] = categories

            # Build one-hot columns deterministically
            created = []
            # Optionally drop_first: drop the first category in sorted order
            categories_to_encode = categories[1:] if drop_first else categories[:]

            for cat in categories_to_encode:
                new_col = f"{col}__{cat}"
                df_after[new_col] = (s_mapped == cat).astype("Int64")
                created.append(new_col)

            diff["created_columns"].extend(created)
            diff["encoded_columns"].append(col)

            # Drop original column
            df_after = df_after.drop(columns=[col])
            diff["dropped_original_columns"].append(col)

    else:  # ordinal
        for col in columns:
            s = df_after[col].astype("string")

            # Categories (excluding NA)
            non_na = s.dropna()
            if non_na.empty:
                warnings.append(f"encode_categorical(ordinal): column '{col}' has only NA values; skipped.")
                continue

            if ordinal_order == "alphabetical":
                categories = sorted(set(non_na.astype(str).tolist()))
            else:
                # frequency: stable order by (-count, category) for deterministic ties
                vc = non_na.value_counts()
                categories = sorted(vc.index.astype(str).tolist(), key=lambda x: (-int(vc[x]), x))

            mapping = {cat: i for i, cat in enumerate(categories)}
            diff["mappings"][col] = mapping

            # Apply mapping; handle unknowns (not very relevant here since mapping derived from data)
            # But kept for consistency if later used with external mapping
            def map_ordinal(x):
                if pd.isna(x):
                    return pd.NA
                x = str(x)
                if x not in mapping:
                    if handle_unknown == "error":
                        raise ValueError(f"encode_categorical(ordinal): unknown category '{x}' in column '{col}'")
                    return pd.NA
                return mapping[x]

            new_col = f"{col}__ord"
            df_after[new_col] = s.map(map_ordinal).astype("Int64")

            diff["created_columns"].append(new_col)
            diff["encoded_columns"].append(col)

            # Drop original
            df_after = df_after.drop(columns=[col])
            diff["dropped_original_columns"].append(col)

    return df_after, diff, warnings


def remove_outliers(
    df: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """
    Deterministically handle outliers using IQR or Z-score.

    params:
      - columns: list[str] (required)
      - method: "iqr" | "zscore" (required)
      - iqr_k: float (optional, default 1.5)        [iqr]
      - z_thresh: float (optional, default 3.0)     [zscore]
      - mode: "drop_rows" | "clip" (optional, default "drop_rows")
      - combine: "any" | "all" (optional, default "any")  # if multiple columns: drop if any/all are outliers
    """

    warnings: List[str] = []
    df_after = df.copy()

    columns = params.get("columns")
    method = params.get("method")
    mode = params.get("mode", "drop_rows")
    combine = params.get("combine", "any")

    if not columns or not isinstance(columns, list):
        raise ValueError("remove_outliers requires 'columns' as a non-empty list")

    if method not in {"iqr", "zscore"}:
        raise ValueError("remove_outliers requires 'method' in {iqr, zscore}")

    if mode not in {"drop_rows", "clip"}:
        raise ValueError("remove_outliers 'mode' must be 'drop_rows' or 'clip'")

    if combine not in {"any", "all"}:
        raise ValueError("remove_outliers 'combine' must be 'any' or 'all'")

    for c in columns:
        if c not in df_after.columns:
            raise ValueError(f"Column '{c}' does not exist")

    # Ensure numeric columns only
    numeric_cols = []
    non_numeric = []
    for c in columns:
        if pd.api.types.is_numeric_dtype(df_after[c]):
            numeric_cols.append(c)
        else:
            non_numeric.append(c)

    if non_numeric:
        raise ValueError(f"remove_outliers only supports numeric columns. Non-numeric: {non_numeric}")

    iqr_k = float(params.get("iqr_k", 1.5))
    z_thresh = float(params.get("z_thresh", 3.0))

    bounds: Dict[str, Dict[str, float]] = {}
    outlier_masks: Dict[str, pd.Series] = {}

    for c in numeric_cols:
        s = df_after[c]

        # Skip columns with all NA
        if s.dropna().empty:
            warnings.append(f"remove_outliers: column '{c}' is all-NA; skipped.")
            outlier_masks[c] = pd.Series([False] * len(df_after), index=df_after.index)
            continue

        if method == "iqr":
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1

            if pd.isna(iqr) or iqr == 0:
                warnings.append(f"remove_outliers(iqr): column '{c}' has IQR=0; no outliers removed.")
                low, high = float(q1), float(q3)
                mask = pd.Series([False] * len(df_after), index=df_after.index)
            else:
                low = float(q1 - iqr_k * iqr)
                high = float(q3 + iqr_k * iqr)
                mask = (s < low) | (s > high)

        else:  # zscore
            mean = s.mean()
            std = s.std(ddof=0)

            if pd.isna(std) or std == 0:
                warnings.append(f"remove_outliers(zscore): column '{c}' has std=0; no outliers removed.")
                low, high = float(s.min()), float(s.max())
                mask = pd.Series([False] * len(df_after), index=df_after.index)
            else:
                z = (s - mean) / std
                mask = z.abs() > z_thresh
                # For audit, approximate bounds from threshold
                low = float(mean - z_thresh * std)
                high = float(mean + z_thresh * std)

        bounds[c] = {"low": low, "high": high}
        outlier_masks[c] = mask.fillna(False)

    # Combine masks across columns
    if len(numeric_cols) == 1:
        combined_mask = outlier_masks[numeric_cols[0]]
    else:
        masks = [outlier_masks[c] for c in numeric_cols]
        combined_mask = masks[0].copy()
        if combine == "any":
            for m in masks[1:]:
                combined_mask = combined_mask | m
        else:
            for m in masks[1:]:
                combined_mask = combined_mask & m

    n_outlier_rows = int(combined_mask.sum())

    diff: Dict[str, Any] = {
        "method": method,
        "mode": mode,
        "combine": combine,
        "columns": numeric_cols,
        "bounds": bounds,
        "rows_flagged_as_outliers": n_outlier_rows,
        "rows_before": int(len(df_after)),
        "rows_after": None,
        "rows_removed": 0,
        "values_clipped_count": {},
    }

    if n_outlier_rows == 0:
        warnings.append("remove_outliers: no outliers detected with the given configuration.")
        diff["rows_after"] = int(len(df_after))
        return df_after, diff, warnings

    if mode == "drop_rows":
        df_after = df_after.loc[~combined_mask]
        diff["rows_after"] = int(len(df_after))
        diff["rows_removed"] = int(diff["rows_before"] - diff["rows_after"])

    else:  # clip
        # Clip each column independently within its bounds; count how many values were clipped
        for c in numeric_cols:
            low = bounds[c]["low"]
            high = bounds[c]["high"]
            s = df_after[c]
            clipped = s.clip(lower=low, upper=high)
            n_clipped = int((clipped != s).sum())
            if n_clipped > 0:
                diff["values_clipped_count"][c] = n_clipped
            df_after[c] = clipped
        diff["rows_after"] = int(len(df_after))
        diff["rows_removed"] = 0

    return df_after, diff, warnings
