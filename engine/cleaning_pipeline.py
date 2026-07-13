"""
Plan-based cleaning transformer - sklearn-compatible.

Cleaning is applied INSIDE cross_validate, fold-by-fold. The plan is decided
once per (dataset, llm-tag); state of stateful actions (medians, MICE, encoder
vocabularies, outlier bounds) is fit on the training fold only.

Stateless actions (fix_column_names, normalize_text, drop_column, cast_type)
are re-applied identically on transform.

`deduplicate` cannot live in a sklearn fit/transform contract (changes row
count) - it is a no-op here and must be applied as a pre-step before CV.

v2 (C4 additions):
- encode_categorical_per_column: per-column encoding (ordinal vs one_hot)
- select_features: drop low-variance / highly-correlated features
- clip_outliers: IQR-based outlier clipping (renamed from remove_outliers for clarity)
- bin_numeric: discretize continuous features into bins
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, KBinsDiscretizer

from engine.actions import fix_column_names, normalize_text
from engine.config import RANDOM_STATE


def _na_to_none(df: pd.DataFrame) -> pd.DataFrame:
    """Replace any pd.NA / NaN-like marker in non-numeric columns by None.
    Required because sklearn SimpleImputer's mask `X != X` cannot evaluate
    pandas pd.NA (raises TypeError: boolean value of NA is ambiguous).
    """
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            continue
        converted: List[Any] = []
        for v in s.tolist():
            try:
                if v is pd.NA or pd.isna(v):
                    converted.append(None)
                    continue
            except Exception:
                pass
            converted.append(v)
        out[c] = pd.Series(converted, index=out.index, dtype=object)
    return out


def _cat_to_imputable(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to object dtype with np.nan for missing values.
    SimpleImputer requires np.nan (not None or pd.NA) to detect missing values
    in object-dtype arrays during transform.
    """
    out = df.copy()
    for c in out.columns:
        # Convert to object dtype to escape StringDtype/pd.NA issues
        vals = []
        for v in out[c].tolist():
            try:
                if v is None or v is pd.NA or pd.isna(v):
                    vals.append(np.nan)
                    continue
            except (TypeError, ValueError):
                pass
            vals.append(v)
        out[c] = pd.Series(vals, index=out.index, dtype=object)
    return out


class PlanBasedCleaner(BaseEstimator, TransformerMixin):
    SUPPORTED = {
        "fix_column_names",
        "normalize_text",
        "drop_column",
        "handle_missing",
        "encode_categorical",
        "encode_categorical_per_column",       # NEW (C4)
        "remove_outliers",
        "clip_outliers",                        # NEW alias (C4)
        "select_features",                      # NEW (C4)
        "bin_numeric",                          # NEW (C4)
        "cast_type",
        "semantic_missing_to_category",         # NEW (C4 Phase B)
        "add_missing_indicators",               # NEW (C4 Phase B)
        "group_rare_categories",                # NEW (C4 v2)
        "transform_numeric_skewed",             # NEW (C4 v2)
        "scale_features",                       # NEW (C4 G12 - StandardScaler for linear models)
    }

    def __init__(self, plan: Optional[Dict[str, Any]] = None, target_column: Optional[str] = None):
        self.plan = plan if plan is not None else {"actions": []}
        self.target_column = target_column

    # ---------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        self._fit_state: List[Dict[str, Any]] = []
        self._skipped: List[str] = []

        for action in self.plan.get("actions", []):
            name = (action or {}).get("action")
            params = dict((action or {}).get("params") or {})
            target_cols = list((action or {}).get("target_columns") or [])

            if name == "deduplicate":
                self._fit_state.append({"name": "deduplicate", "noop": True})
                continue

            # Alias: clip_outliers → remove_outliers handler
            if name == "clip_outliers":
                name = "remove_outliers"

            if name not in self.SUPPORTED and name != "remove_outliers":
                self._skipped.append(name or "<missing>")
                self._fit_state.append({"name": name, "skipped": True})
                continue

            handler = getattr(self, f"_fit_{name}", None)
            if handler is None:
                self._skipped.append(name or "<missing>")
                self._fit_state.append({"name": name, "skipped": True})
                continue

            df, state = handler(df, params, target_cols)
            state["name"] = name
            self._fit_state.append(state)

        df = _na_to_none(df)
        self._fit_columns_: List[str] = list(df.columns)
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for state in self._fit_state:
            name = state.get("name")
            if state.get("noop") or state.get("skipped"):
                continue
            handler = getattr(self, f"_transform_{name}", None)
            if handler is None:
                continue
            df = handler(df, state)

        for c in self._fit_columns_:
            if c not in df.columns:
                df[c] = 0
        df = df.reindex(columns=self._fit_columns_)
        df = _na_to_none(df)
        return df

    # ============================ Per-action handlers ====================

    # --- fix_column_names (stateless) ---
    def _fit_fix_column_names(self, df, params, target_cols):
        df_after, _, _ = fix_column_names(df)
        return df_after, {}

    def _transform_fix_column_names(self, df, state):
        df_after, _, _ = fix_column_names(df)
        return df_after

    # --- normalize_text (stateless) ---
    def _fit_normalize_text(self, df, params, target_cols):
        cols = self._resolve_string_columns(df, params, target_cols)
        ops = params.get("ops", ["strip", "collapse_whitespace"])
        null_like = params.get("null_like_to_nan", [])
        replace_rules = params.get("replace", [])
        if cols:
            full_params = {"columns": cols, "ops": ops}
            if null_like:
                full_params["null_like_to_nan"] = null_like
            if replace_rules:
                full_params["replace"] = replace_rules
            df, _, _ = normalize_text(df, full_params)
        return df, {"cols": cols, "ops": ops, "null_like_to_nan": null_like,
                     "replace": replace_rules}

    def _transform_normalize_text(self, df, state):
        cols = [c for c in state.get("cols", []) if c in df.columns]
        if cols:
            full_params = {"columns": cols, "ops": state.get("ops")}
            null_like = state.get("null_like_to_nan", [])
            replace_rules = state.get("replace", [])
            if null_like:
                full_params["null_like_to_nan"] = null_like
            if replace_rules:
                full_params["replace"] = replace_rules
            df, _, _ = normalize_text(df, full_params)
        return df

    # --- drop_column (stateless) ---
    def _fit_drop_column(self, df, params, target_cols):
        cols_in = params.get("columns") or target_cols or []
        cols = [c for c in cols_in if c in df.columns and c != self.target_column]
        df = df.drop(columns=cols, errors="ignore")
        return df, {"cols": cols}

    def _transform_drop_column(self, df, state):
        cols = [c for c in state.get("cols", []) if c in df.columns]
        return df.drop(columns=cols, errors="ignore")

    # --- cast_type (stateless) ---
    def _fit_cast_type(self, df, params, target_cols):
        from engine.actions import cast_type
        cols = params.get("columns") or target_cols
        if cols:
            df, _, _ = cast_type(df, {**params, "columns": cols})
        return df, {"params": {**params, "columns": cols}}

    def _transform_cast_type(self, df, state):
        from engine.actions import cast_type
        params = state.get("params") or {}
        cols = [c for c in (params.get("columns") or []) if c in df.columns]
        if cols:
            df, _, _ = cast_type(df, {**params, "columns": cols})
        return df

    # --- handle_missing - STATEFUL ---
    def _fit_handle_missing(self, df, params, target_cols):
        cols_in = params.get("columns") or target_cols or [
            c for c in df.columns
            if c != self.target_column and df[c].isna().any()
        ]
        cols_in = [c for c in cols_in if c in df.columns and c != self.target_column]

        num_cols = [c for c in cols_in if pd.api.types.is_numeric_dtype(df[c])
                    and not pd.api.types.is_bool_dtype(df[c])]
        cat_cols = [c for c in cols_in if c not in num_cols]

        # Model-aware categorical imputation strategy:
        # "preserve_missing" (default): _na_to_none → OHE creates binary
        #   missingness-indicator column. Best for distance-based (KNN) and
        #   tree-based (RF, GBM) models that can exploit the signal.
        # "fill_mode": _cat_to_imputable → SimpleImputer replaces NaN with
        #   the mode, eliminating the extra feature. Best for linear models
        #   (LogReg, Ridge) where extra low-information features add noise.
        cat_missing_strategy = params.get("cat_missing_strategy", "preserve_missing")

        # G8 (enforce_c4_v3): strategy='median' is injected for tree-based models
        # (RF, GBM) because MICE over-smooths numeric features and degrades
        # tree performance relative to the C0 median baseline.
        # Linear/KNN models keep MICE (strategy='impute') for better estimates.
        num_strategy = params.get("strategy", "impute")

        num_imputer = None
        cat_imputer = None

        if num_cols:
            if num_strategy == "median":
                num_imputer = SimpleImputer(strategy="median")
            else:
                num_imputer = IterativeImputer(max_iter=10, random_state=RANDOM_STATE)
            num_imputer.fit(df[num_cols])
            df[num_cols] = num_imputer.transform(df[num_cols])

        if cat_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            if cat_missing_strategy == "fill_mode":
                # Standard imputation: NaN → np.nan → SimpleImputer fills with mode
                cat_block = _cat_to_imputable(df[cat_cols])
                cat_imputer.fit(cat_block)
                df[cat_cols] = cat_imputer.transform(cat_block)
            else:
                # Preserve missingness: NaN → None → SimpleImputer leaves None
                # → OHE creates binary "was_missing" column (extra feature)
                cat_block = _na_to_none(df[cat_cols])
                cat_imputer.fit(cat_block)
                df[cat_cols] = cat_imputer.transform(cat_block)

        return df, {
            "num_cols": num_cols, "num_imputer": num_imputer,
            "cat_cols": cat_cols, "cat_imputer": cat_imputer,
            "cat_missing_strategy": cat_missing_strategy,
            "num_strategy": num_strategy,
        }

    def _transform_handle_missing(self, df, state):
        num_cols = [c for c in state.get("num_cols", []) if c in df.columns]
        cat_cols = [c for c in state.get("cat_cols", []) if c in df.columns]
        cat_missing_strategy = state.get("cat_missing_strategy", "preserve_missing")
        if num_cols and state.get("num_imputer") is not None:
            df[num_cols] = state["num_imputer"].transform(df[num_cols])
        if cat_cols and state.get("cat_imputer") is not None:
            if cat_missing_strategy == "fill_mode":
                cat_block = _cat_to_imputable(df[cat_cols])
            else:
                cat_block = _na_to_none(df[cat_cols])
            df[cat_cols] = state["cat_imputer"].transform(cat_block)
        return df

    # --- encode_categorical - STATEFUL (global method, legacy) ---
    def _fit_encode_categorical(self, df, params, target_cols):
        method = params.get("method", "one_hot")
        cols_in = params.get("columns") or target_cols or [
            c for c in df.columns
            if c != self.target_column
            and (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]))
        ]
        cols = [c for c in cols_in if c in df.columns and c != self.target_column]
        if not cols:
            return df, {"cols": [], "encoder": None, "method": method, "new_columns": []}

        df[cols] = df[cols].astype(str)

        if method == "ordinal":
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            enc.fit(df[cols])
            arr = enc.transform(df[cols])
            new_cols = [f"{c}__ord" for c in cols]
        else:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            enc.fit(df[cols])
            arr = enc.transform(df[cols])
            new_cols = list(enc.get_feature_names_out(cols))

        enc_df = pd.DataFrame(arr, columns=new_cols, index=df.index)
        df = df.drop(columns=cols).join(enc_df)
        return df, {"cols": cols, "encoder": enc, "method": method, "new_columns": new_cols}

    def _transform_encode_categorical(self, df, state):
        cols = state.get("cols") or []
        enc = state.get("encoder")
        if not cols or enc is None:
            return df

        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df[cols] = df[cols].astype(str)
        arr = enc.transform(df[cols])
        new_cols = state.get("new_columns") or []
        if state.get("method") == "ordinal":
            new_cols = new_cols or [f"{c}__ord" for c in cols]
        enc_df = pd.DataFrame(arr, columns=new_cols, index=df.index)
        df = df.drop(columns=[c for c in cols if c in df.columns]).join(enc_df)
        return df

    # =====================================================================
    # NEW (C4): encode_categorical_per_column - per-column encoding
    # =====================================================================
    def _fit_encode_categorical_per_column(self, df, params, target_cols):
        """
        Per-column encoding: each column gets its own method (ordinal or one_hot).

        params.column_encodings: dict mapping column_name → "ordinal" | "one_hot"
        params.default_method: fallback for columns not in column_encodings
        """
        column_encodings_raw = params.get("column_encodings", {})
        default_method = params.get("default_method", "one_hot")

        # GUARDRAIL: normalise column_encodings keys to match the actual DataFrame
        # column names. fix_column_names lowercases all column names, but the LLM
        # plan may have been generated before that step (or from the original profile
        # which had mixed-case names, e.g. house_prices: 'MSZoning', 'LotShape').
        # Build a lookup from normalised name → actual df column name.
        df_col_lower = {c.strip().lower().replace(" ", "_").replace("-", "_"): c for c in df.columns}
        column_encodings = {}
        for k, v in column_encodings_raw.items():
            normalised = k.strip().lower().replace(" ", "_").replace("-", "_")
            actual = df_col_lower.get(normalised, k)  # map to real col name, fallback to original
            column_encodings[actual] = v

        # GUARDRAIL: also include numeric columns that are explicitly listed in
        # column_encodings - these are integer-encoded categoricals (e.g. heart's
        # cp, thal, restecg) that the LLM/context correctly identified as nominal
        # or ordinal. Without this, datasets where all columns are numeric (float64)
        # would silently skip encoding and produce C4 == C0.
        explicitly_listed = set(column_encodings.keys())

        all_cat_cols = [
            c for c in df.columns
            if c != self.target_column
            and (
                pd.api.types.is_object_dtype(df[c])
                or pd.api.types.is_string_dtype(df[c])
                or c in explicitly_listed  # include numeric cols marked by context
            )
        ]
        if target_cols:
            all_cat_cols = [
                c for c in target_cols
                if c in df.columns and c != self.target_column
                and (
                    pd.api.types.is_object_dtype(df[c])
                    or pd.api.types.is_string_dtype(df[c])
                    or c in explicitly_listed
                )
            ]

        # Restrict to columns that are actually present
        all_cat_cols = [c for c in all_cat_cols if c in df.columns]

        if not all_cat_cols:
            return df, {"encoders": [], "new_columns_map": {}}

        # Split into ordinal and one-hot groups
        ordinal_cols = [c for c in all_cat_cols
                       if column_encodings.get(c, default_method) == "ordinal"]
        onehot_cols = [c for c in all_cat_cols
                      if column_encodings.get(c, default_method) == "one_hot"]

        encoders = []
        new_columns_map = {}

        # Encode ordinal columns
        if ordinal_cols:
            df[ordinal_cols] = df[ordinal_cols].astype(str)
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            enc.fit(df[ordinal_cols])
            arr = enc.transform(df[ordinal_cols])
            new_cols = [f"{c}__ord" for c in ordinal_cols]
            enc_df = pd.DataFrame(arr, columns=new_cols, index=df.index)
            df = df.drop(columns=ordinal_cols).join(enc_df)
            encoders.append({"method": "ordinal", "cols": ordinal_cols, "encoder": enc, "new_cols": new_cols})
            for c, nc in zip(ordinal_cols, new_cols):
                new_columns_map[c] = [nc]

        # Encode one-hot columns
        if onehot_cols:
            df[onehot_cols] = df[onehot_cols].astype(str)
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            enc.fit(df[onehot_cols])
            arr = enc.transform(df[onehot_cols])
            new_cols = list(enc.get_feature_names_out(onehot_cols))
            enc_df = pd.DataFrame(arr, columns=new_cols, index=df.index)
            df = df.drop(columns=onehot_cols).join(enc_df)
            encoders.append({"method": "one_hot", "cols": onehot_cols, "encoder": enc, "new_cols": new_cols})
            # Map original cols to new cols for reference
            for c in onehot_cols:
                new_columns_map[c] = [nc for nc in new_cols if nc.startswith(f"{c}_")]

        return df, {"encoders": encoders, "new_columns_map": new_columns_map}

    def _transform_encode_categorical_per_column(self, df, state):
        for enc_info in state.get("encoders", []):
            cols = enc_info["cols"]
            enc = enc_info["encoder"]
            new_cols = enc_info["new_cols"]
            method = enc_info["method"]

            existing_cols = [c for c in cols if c in df.columns]
            missing_cols = [c for c in cols if c not in df.columns]

            if not existing_cols and not missing_cols:
                continue

            # Add missing columns with empty string
            for c in missing_cols:
                if c not in df.columns:
                    df[c] = ""

            all_cols = existing_cols + missing_cols
            df[cols] = df[cols].astype(str)
            arr = enc.transform(df[cols])
            enc_df = pd.DataFrame(arr, columns=new_cols, index=df.index)
            df = df.drop(columns=[c for c in cols if c in df.columns]).join(enc_df)

        return df

    # =====================================================================
    # NEW (C4): select_features - variance + correlation filter
    # =====================================================================
    def _fit_select_features(self, df, params, target_cols):
        """
        Feature selection based on:
        - variance_threshold: drop numeric cols with variance < threshold (default 0.01)
        - correlation_threshold: drop one of two cols if |corr| > threshold (default 0.95)
        - drop_columns: explicit list of columns to drop (LLM-suggested)
        """
        variance_threshold = float(params.get("variance_threshold", 0.01))
        correlation_threshold = float(params.get("correlation_threshold", 0.95))
        explicit_drops = list(params.get("drop_columns", []))

        # 1. Drop explicitly named columns
        explicit_drops = [c for c in explicit_drops if c in df.columns and c != self.target_column]
        if explicit_drops:
            df = df.drop(columns=explicit_drops, errors="ignore")

        # 2. Drop low-variance numeric columns
        num_cols = [c for c in df.columns
                   if c != self.target_column
                   and pd.api.types.is_numeric_dtype(df[c])
                   and not pd.api.types.is_bool_dtype(df[c])]

        low_var_drops = []
        if num_cols and variance_threshold > 0:
            variances = df[num_cols].var()
            low_var_drops = [c for c in num_cols if variances.get(c, 1) < variance_threshold]
            if low_var_drops:
                df = df.drop(columns=low_var_drops, errors="ignore")
                num_cols = [c for c in num_cols if c not in low_var_drops]

        # 3. Drop highly correlated features (keep first, drop second)
        corr_drops = []
        if len(num_cols) >= 2 and correlation_threshold < 1.0:
            try:
                corr_matrix = df[num_cols].corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                corr_drops = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
                if corr_drops:
                    df = df.drop(columns=corr_drops, errors="ignore")
            except Exception:
                corr_drops = []

        all_drops = explicit_drops + low_var_drops + corr_drops
        return df, {"dropped": all_drops, "variance_threshold": variance_threshold,
                    "correlation_threshold": correlation_threshold}

    def _transform_select_features(self, df, state):
        dropped = [c for c in state.get("dropped", []) if c in df.columns]
        if dropped:
            df = df.drop(columns=dropped, errors="ignore")
        return df

    # =====================================================================
    # NEW (C4): bin_numeric - discretize continuous features
    # =====================================================================
    def _fit_bin_numeric(self, df, params, target_cols):
        """
        Bin continuous features into discrete buckets.

        params.columns: list of columns to bin (or auto-detect if empty)
        params.n_bins: number of bins (default 5)
        params.strategy: 'quantile', 'uniform', or 'kmeans' (default 'quantile')
        params.encode_bins: 'ordinal' (default) or 'onehot'
        """
        cols_in = params.get("columns") or target_cols or []
        n_bins = int(params.get("n_bins", 5))
        strategy = params.get("strategy", "quantile")
        encode_bins = params.get("encode_bins", "ordinal")

        # Filter to numeric columns that exist
        cols = [c for c in cols_in if c in df.columns
                and c != self.target_column
                and pd.api.types.is_numeric_dtype(df[c])]

        if not cols:
            return df, {"cols": [], "binner": None, "new_cols": []}

        # Fill NaN before binning (use median)
        for c in cols:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())

        sklearn_encode = "ordinal" if encode_bins == "ordinal" else "onehot-dense"
        binner = KBinsDiscretizer(n_bins=n_bins, encode=sklearn_encode,
                                  strategy=strategy, subsample=None)

        try:
            binner.fit(df[cols])
            arr = binner.transform(df[cols])

            if encode_bins == "ordinal":
                new_cols = [f"{c}__bin" for c in cols]
            else:
                new_cols = [f"{c}__bin_{i}" for c in cols for i in range(n_bins)]

            if arr.shape[1] != len(new_cols):
                # Adjust if onehot produced different number of columns
                new_cols = [f"bin_{i}" for i in range(arr.shape[1])]

            enc_df = pd.DataFrame(arr, columns=new_cols, index=df.index)
            df = df.drop(columns=cols).join(enc_df)
        except Exception:
            # If binning fails (e.g., not enough unique values), keep originals
            return df, {"cols": [], "binner": None, "new_cols": []}

        return df, {"cols": cols, "binner": binner, "new_cols": new_cols, "encode_bins": encode_bins}

    def _transform_bin_numeric(self, df, state):
        cols = state.get("cols", [])
        binner = state.get("binner")
        new_cols = state.get("new_cols", [])

        if not cols or binner is None:
            return df

        existing_cols = [c for c in cols if c in df.columns]
        if len(existing_cols) != len(cols):
            return df  # Schema mismatch, skip

        for c in cols:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())

        try:
            arr = binner.transform(df[cols])
            enc_df = pd.DataFrame(arr, columns=new_cols, index=df.index)
            df = df.drop(columns=cols).join(enc_df)
        except Exception:
            pass  # Keep original columns on failure

        return df

    # --- remove_outliers - STATEFUL (clip mode only) ---
    def _fit_remove_outliers(self, df, params, target_cols):
        method = params.get("method", "iqr")
        k = float(params.get("iqr_k", 3.0))
        z_thresh = float(params.get("z_thresh", 3.0))
        cols_in = params.get("columns") or target_cols or [
            c for c in df.columns
            if c != self.target_column and pd.api.types.is_numeric_dtype(df[c])
        ]
        cols = [c for c in cols_in if c in df.columns
                and pd.api.types.is_numeric_dtype(df[c])
                and c != self.target_column]

        bounds: Dict[str, Any] = {}
        for c in cols:
            s = df[c].dropna()
            if s.empty:
                continue
            if method == "zscore":
                mu, sd = float(s.mean()), float(s.std(ddof=0) or 0)
                if sd == 0:
                    continue
                bounds[c] = (mu - z_thresh * sd, mu + z_thresh * sd)
            else:
                q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
                iqr = q3 - q1
                if iqr == 0:
                    continue
                bounds[c] = (q1 - k * iqr, q3 + k * iqr)

        for c, (lo, hi) in bounds.items():
            df[c] = df[c].clip(lower=lo, upper=hi)

        return df, {"bounds": bounds}

    def _transform_remove_outliers(self, df, state):
        for c, (lo, hi) in (state.get("bounds") or {}).items():
            if c in df.columns:
                df[c] = df[c].clip(lower=lo, upper=hi)
        return df

    # --- semantic_missing_to_category --- STATEFUL ---
    def _fit_semantic_missing_to_category(self, df, params, target_cols):
        """
        Replace NA in categorical columns with a meaningful label where NA
        genuinely means "this feature is absent" (e.g. PoolQC=NA -> 'No_Pool').

        params:
          fill_value  : str, default "None"
          column_fills: dict {col: fill_str}  per-column overrides (bypass min_na_rate)
          min_na_rate : float, default 0.0    auto-fill cols with na_rate >= this
        """
        fill_value  = params.get("fill_value", "None")
        col_fills   = params.get("column_fills", {})
        min_na_rate = float(params.get("min_na_rate", 0.0))

        # Normalise col_fills keys to lowercase (post fix_column_names)
        df_col_lower = {c.strip().lower().replace(" ", "_").replace("-", "_"): c
                        for c in df.columns}
        col_fills_norm: Dict[str, str] = {}
        for k, v in col_fills.items():
            normalised = k.strip().lower().replace(" ", "_").replace("-", "_")
            actual = df_col_lower.get(normalised, normalised)
            if actual in df.columns:
                col_fills_norm[actual] = v

        n = len(df)
        mapping: Dict[str, str] = {}
        for c in df.columns:
            if c == self.target_column:
                continue
            if not (pd.api.types.is_object_dtype(df[c]) or
                    pd.api.types.is_string_dtype(df[c])):
                continue
            na_count = int(df[c].isna().sum())
            if c in col_fills_norm:
                mapping[c] = col_fills_norm[c]
            elif na_count > 0 and (n == 0 or na_count / n >= min_na_rate):
                mapping[c] = fill_value

        for c, val in mapping.items():
            df[c] = df[c].fillna(val)

        return df, {"mapping": mapping}

    def _transform_semantic_missing_to_category(self, df, state):
        for c, val in (state.get("mapping") or {}).items():
            if c in df.columns:
                df[c] = df[c].fillna(val)
        return df

    # --- add_missing_indicators --- STATEFUL ---
    def _fit_add_missing_indicators(self, df, params, target_cols):
        """
        Add binary indicator columns (<col>_was_missing = 1/0) before imputation
        to preserve the missingness signal.

        params:
          min_na_rate     : float, default 0.01
          max_indicators  : int,   default 20
          columns         : list   explicit columns (bypasses min_na_rate)
          exclude_columns : list   never add indicators for these
        """
        min_na_rate    = float(params.get("min_na_rate", 0.01))
        max_indicators = int(params.get("max_indicators", 20))
        explicit_cols  = list(params.get("columns") or [])
        exclude_cols   = set(params.get("exclude_columns") or [])

        n = len(df)
        df_col_lower = {c.strip().lower().replace(" ", "_").replace("-", "_"): c
                        for c in df.columns}

        if explicit_cols:
            candidate_cols = []
            for k in explicit_cols:
                norm = k.strip().lower().replace(" ", "_").replace("-", "_")
                actual = df_col_lower.get(norm, norm)
                if actual in df.columns and actual not in exclude_cols:
                    candidate_cols.append(actual)
        else:
            candidate_cols = [
                c for c in df.columns
                if c != self.target_column
                and c not in exclude_cols
                and df[c].isna().sum() > 0
                and (n == 0 or df[c].isna().sum() / n >= min_na_rate)
            ]

        indicator_cols: List[str] = []
        for c in candidate_cols[:max_indicators]:
            ind_name = f"{c}_was_missing"
            df[ind_name] = df[c].isna().astype(int)
            indicator_cols.append(ind_name)

        return df, {"indicator_cols": indicator_cols}

    def _transform_add_missing_indicators(self, df, state):
        for c_ind in (state.get("indicator_cols") or []):
            src = c_ind[:-len("_was_missing")]
            if src in df.columns:
                df[c_ind] = df[src].isna().astype(int)
            elif c_ind not in df.columns:
                df[c_ind] = 0
        return df

    # --- group_rare_categories --- STATEFUL ---
    def _fit_group_rare_categories(self, df, params, target_cols):
        """
        Merge infrequent category values into a single 'Other' label.
        Reduces OHE dimensionality and sparse dummy columns that hurt
        linear models and KNN.

        params:
          min_frequency_pct : float, default 0.01  (1% of train rows)
          min_frequency_abs : int,   default 5
          max_categories    : int,   default None  keep top-N, rest → Other
          replacement_label : str,   default "Other"
          columns           : list   explicit columns (default: all object columns)
        """
        min_pct    = float(params.get("min_frequency_pct", 0.01))
        min_abs    = int(params.get("min_frequency_abs", 5))
        max_cats   = params.get("max_categories", None)
        label      = params.get("replacement_label", "Other")
        explicit   = list(params.get("columns") or [])

        n = len(df)
        min_count = max(min_abs, int(round(min_pct * n)))

        if explicit:
            cols = [c for c in explicit if c in df.columns]
        else:
            cols = [
                c for c in df.columns
                if c != self.target_column
                and (pd.api.types.is_object_dtype(df[c]) or
                     pd.api.types.is_string_dtype(df[c]))
            ]

        keep_map: Dict[str, set] = {}
        for c in cols:
            vc = df[c].value_counts(dropna=False)
            # Keep categories that meet the frequency threshold
            frequent = set(vc[vc >= min_count].index.dropna())
            if max_cats is not None:
                # Keep only top-N by frequency
                top_n = set(vc.head(int(max_cats)).index.dropna())
                frequent = frequent & top_n
            keep_map[c] = frequent
            df[c] = df[c].apply(lambda v: v if (pd.notna(v) and v in frequent) else
                                  (label if pd.notna(v) else v))

        return df, {"keep_map": {c: list(v) for c, v in keep_map.items()},
                    "label": label}

    def _transform_group_rare_categories(self, df, state):
        label    = state.get("label", "Other")
        keep_map = state.get("keep_map") or {}
        for c, keep_list in keep_map.items():
            if c not in df.columns:
                continue
            keep_set = set(keep_list)
            df[c] = df[c].apply(lambda v: v if (pd.notna(v) and v in keep_set) else
                                  (label if pd.notna(v) else v))
        return df

    # --- transform_numeric_skewed --- STATEFUL ---
    def _fit_transform_numeric_skewed(self, df, params, target_cols):
        """
        Apply log1p (or Yeo-Johnson) to right-skewed numeric columns.
        Reduces the influence of extreme values on distance-based and
        linear models. Should only be applied for logreg/ridge/knn
        (skip for RF/GBM via model_family param).

        params:
          skewness_threshold : float, default 1.0   apply if |skew| > threshold
          method             : str,   default "log1p"  or "yeo-johnson"
          columns            : list   explicit columns (default: auto-detect)
          exclude_columns    : list   never transform these
          model_family       : str    if "tree", skip entirely
        """
        import scipy.stats as stats

        model_family = params.get("model_family", "linear")
        if model_family == "tree":
            return df, {"transforms": {}, "skipped_tree": True}

        thresh     = float(params.get("skewness_threshold", 1.0))
        method     = params.get("method", "log1p")
        explicit   = list(params.get("columns") or [])
        exclude    = set(params.get("exclude_columns") or [])

        num_cols = [
            c for c in df.columns
            if c != self.target_column
            and c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].nunique() > 10          # skip binary / integer-coded categoricals
        ]

        if explicit:
            num_cols = [c for c in explicit if c in df.columns and c not in exclude]

        transforms: Dict[str, Any] = {}
        for c in num_cols:
            col = df[c].dropna()
            if len(col) < 10:
                continue
            skew = float(col.skew())
            if abs(skew) <= thresh:
                continue
            if method == "log1p":
                if col.min() < 0:
                    continue  # log1p requires non-negative
                df[c] = np.log1p(df[c].clip(lower=0))
                transforms[c] = {"method": "log1p", "skew_before": skew}
            elif method == "yeo-johnson":
                from sklearn.preprocessing import PowerTransformer
                pt = PowerTransformer(method="yeo-johnson", standardize=False)
                vals = df[c].values.astype(float)
                mask = ~np.isnan(vals)
                pt.fit(vals[mask].reshape(-1, 1))
                out = vals.copy()
                out[mask] = pt.transform(vals[mask].reshape(-1, 1)).ravel()
                df[c] = out
                transforms[c] = {"method": "yeo-johnson", "skew_before": skew,
                                  "_pt": pt}

        return df, {"transforms": transforms}

    def _transform_transform_numeric_skewed(self, df, state):
        if state.get("skipped_tree"):
            return df
        for c, info in (state.get("transforms") or {}).items():
            if c not in df.columns:
                continue
            if info["method"] == "log1p":
                df[c] = np.log1p(df[c].clip(lower=0))
            elif info["method"] == "yeo-johnson":
                pt = info.get("_pt")
                if pt is None:
                    continue
                vals = df[c].values.astype(float)
                mask = ~np.isnan(vals)
                if mask.any():
                    vals[mask] = pt.transform(vals[mask].reshape(-1, 1)).ravel()
                df[c] = vals
        return df

    # --- scale_features - STATEFUL (G12: StandardScaler for linear models) ---
    def _fit_scale_features(self, df, params, target_cols):
        """
        Apply StandardScaler to all numeric features.
        Only meaningful for linear/distance-based models (Ridge, LogReg, KNN).
        Tree models skip this action via G1/G12 guardrail injection logic.

        params:
          columns        : list  explicit columns (default: all numeric non-target)
          exclude_columns: list  never scale these
        """
        from sklearn.preprocessing import StandardScaler as _SS
        exclude = set(params.get("exclude_columns") or [])
        explicit = list(params.get("columns") or [])
        num_cols = [
            c for c in df.columns
            if c != self.target_column
            and c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if explicit:
            num_cols = [c for c in explicit if c in df.columns and c not in exclude]

        sc = _SS()
        if num_cols:
            sc.fit(df[num_cols].fillna(0))
        return df, {"scaler": sc, "num_cols": num_cols}

    def _transform_scale_features(self, df, state):
        from sklearn.preprocessing import StandardScaler as _SS
        sc: _SS = state.get("scaler")
        num_cols: list = state.get("num_cols", [])
        cols_present = [c for c in num_cols if c in df.columns]
        if sc is not None and cols_present:
            df[cols_present] = sc.transform(df[cols_present].fillna(0))
        return df

    # ============================ Helpers ===============================
    def _resolve_string_columns(self, df, params, target_cols):
        cols = params.get("columns") or target_cols or [
            c for c in df.columns
            if c != self.target_column
            and (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]))
        ]
        return [c for c in cols if c in df.columns]


# =====================================================================
# Deterministic plan helpers (used by run_experiments.py)
# =====================================================================

def build_c0_empty_plan():
    """C0: no cleaning. Returns an empty plan dict."""
    return {"actions": []}


def build_c1_deterministic_plan():
    """
    C1: fixed deterministic pipeline applied to ALL datasets equally.

    Steps:
      1. fix_column_names
      2. handle_missing  (impute: median for numeric, most_frequent for categorical)
      3. deduplicate
      4. normalize_text  (strip + collapse_whitespace on string columns)
    """
    return {
        "actions": [
            {
                "action": "fix_column_names",
                "rationale": "Standardise column names (lowercase, strip, underscores).",
                "target_columns": [],
                "params": {},
            },
            {
                "action": "handle_missing",
                "rationale": "Impute missing values: median for numeric, most_frequent for categorical.",
                "target_columns": [],
                "params": {"strategy": "impute"},
            },
            {
                "action": "deduplicate",
                "rationale": "Remove exact duplicate rows.",
                "target_columns": [],
                "params": {},
            },
            {
                "action": "normalize_text",
                "rationale": "Strip leading/trailing whitespace; collapse internal whitespace.",
                "target_columns": [],
                "params": {},
            },
        ]
    }
