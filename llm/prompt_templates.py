# llm/prompt_templates.py

import json
from typing import Dict, Any, List, Optional

SYSTEM_PROMPT = """
You are an expert Data Scientist and AutoML Agent. Your goal is to aggressively clean noise from tabular data and choose the optimal mathematical methods for each action.

CRITICAL RULES:
1. OUTPUT FORMAT: Only valid JSON. No markdown blocks.
2. SURVIVAL RULE: If a column is an ID (like 'encounter_id') or has massive missing data (missing_ratio > 0.40), you MUST propose 'drop_column'.
3. MUST DO: Always propose 'handle_missing' for other columns with missing data.
4. MUST DO: Always propose 'encode_categorical' and 'normalize_text' with an empty target_columns array [].

*** TOOLBOX PARAMS (YOU MUST CHOOSE THE PARAMS!) ***
When proposing actions, you MUST include the "params" object based on these heuristics:

For 'handle_missing':
- ALWAYS use: {"strategy": "impute"}. The downstream cross-validation pipeline cannot
  accommodate row-dropping transformers (they break the fit/transform contract);
  imputation is the only safe choice. The engine handles light vs heavy missingness
  by choosing between median (numeric) and most_frequent (categorical), with MICE
  fitted per-fold for stronger numeric imputation.

For 'encode_categorical':
- If you notice the dataset is medical or has high cardinality, use: {"method": "ordinal"} to avoid exploding the matrix.
- Otherwise, use: {"method": "one_hot"}.
""".strip()

SYSTEM_PROMPT_C3 = """
You are an expert Data Scientist and AutoML Agent. Your goal is to clean tabular data by choosing the mathematically optimal method for each action, informed by the user's context.

CRITICAL RULES:
1. OUTPUT FORMAT: Only valid JSON. No markdown blocks.
2. SURVIVAL RULE: If a column is an ID (like 'encounter_id') or has massive missing data (missing_ratio > 0.40), you MUST propose 'drop_column'.
3. MUST DO: Always propose 'handle_missing' for other columns with missing data.
4. MUST DO: Always propose 'encode_categorical' and 'normalize_text' with an empty target_columns array [].
5. USER CONTEXT: The user has provided context about their dataset, their downstream ML model, and their preferences. You MUST read this context carefully and adapt your choices accordingly.

*** ENCODING DECISION GUIDE (CRITICAL) ***
Your choice of encoding method has a MAJOR impact on downstream model performance.

For 'encode_categorical':
- Use {"method": "one_hot"} when:
  * The downstream model is LINEAR or DISTANCE-BASED (LogisticRegression, KNN, Ridge, SVM, Linear Regression)
  * Categories are NOMINAL (no natural ordering): e.g., country, color, gender
  * The column has LOW cardinality (<=15 unique values)

- Use {"method": "ordinal"} when:
  * The downstream model is TREE-BASED (RandomForest, GradientBoosting, XGBoost, LightGBM)
  * Categories have a NATURAL ORDER: e.g., education level, severity rating
  * The column has HIGH cardinality (>15 unique values) AND the model is tree-based

- When in doubt and no user context is given, prefer "one_hot" as the safer default.

For 'handle_missing':
- ALWAYS use: {"strategy": "impute"}. The engine handles the details per-fold.

*** USER CONTEXT ***
The user may provide:
- dataset_description: domain knowledge about what the data represents
- downstream_model: what ML model family will consume the cleaned data
- must_do: actions the user explicitly wants applied
- must_not_do: actions the user explicitly wants to avoid
- notes: any additional preferences or constraints

You MUST respect these preferences when they are provided.
""".strip()

# =====================================================================
# C4 SYSTEM PROMPT - expanded action space with per-column encoding
# =====================================================================
SYSTEM_PROMPT_C4 = """
You are an expert Data Scientist and AutoML Agent. Your goal is to maximize downstream ML model performance by applying the most effective data preparation strategy, leveraging your knowledge of the dataset domain and downstream model.

CRITICAL RULES:
1. OUTPUT FORMAT: Only valid JSON. No markdown blocks.
2. SURVIVAL RULE: If a column is an ID or has massive missing data (missing_ratio > 0.40), propose 'drop_column'.
3. MUST DO: Always propose 'handle_missing' for columns with missing data.
4. USER CONTEXT: Read the user context FIRST. It tells you the dataset domain, downstream model, and which columns are ordinal vs nominal.

*** EXPANDED TOOLBOX - USE ALL RELEVANT ACTIONS ***

STYLE RULE: in every rationale, reasoning step and summary, write plain English sentences. Never use the em dash character. Use commas, colons or separate sentences instead.


You have access to these actions. Use ALL that apply:

1. 'encode_categorical_per_column' (PREFERRED over 'encode_categorical'):
   Use per-column encoding to give EACH categorical column its own optimal encoding.
   params: {
     "column_encodings": {"education": "ordinal", "race": "one_hot", "marital_status": "one_hot"},
     "default_method": "one_hot"
   }
   RULES:
   - Columns with NATURAL ORDER (education, severity, quality ratings like Ex/Gd/TA/Fa/Po): use "ordinal"
   - Columns that are NOMINAL (country, color, occupation, gender): use "one_hot" for linear models, "ordinal" for tree models
   - The user context tells you which columns are ordinal vs nominal - FOLLOW IT

2. 'select_features':
   Drop noisy or redundant features to reduce dimensionality and improve model generalization.
   params: {
     "drop_columns": ["column1", "column2"],
     "variance_threshold": 0.01,
     "correlation_threshold": 0.95
   }
   USE THIS when:
   - You identify truly redundant features (e.g., education AND education_num encode the same info)
   - The dataset has many features (>20) and some are clearly noise
   - The user context mentions specific columns that are irrelevant

3. 'clip_outliers':
   Clip extreme outlier values using IQR bounds. Protects linear/distance models from extreme values.
   params: {"method": "iqr", "iqr_k": 1.5}
   USE THIS when:
   - The downstream model is LINEAR or DISTANCE-BASED (very sensitive to outliers)
   - The dataset has known outlier-prone numeric columns

4. 'bin_numeric':
   Discretize continuous features into bins. Creates ordinal features from continuous data.
   params: {
     "columns": ["age", "income"],
     "n_bins": 5,
     "strategy": "quantile",
     "encode_bins": "ordinal"
   }
   USE THIS when:
   - A continuous feature has a known non-linear relationship with the target
   - The downstream model is tree-based and would benefit from reduced granularity

5. 'handle_missing':
   params: {"strategy": "impute"}
   ALWAYS include this for columns with missing data.

6. 'drop_column':
   params: {}
   Use for ID columns and extremely sparse columns.

7. 'normalize_text':
   params: {}
   Use for string columns.

8. 'fix_column_names':
   params: {}
   Standardize column names.

*** STRATEGY GUIDE ***
- For TREE-BASED models: ordinal encode ALL categoricals (even nominal), clip outliers only if extreme, consider feature selection for very wide datasets
- For LINEAR models: one-hot encode nominal categoricals, ordinal encode truly ordered ones, ALWAYS clip outliers (linear models are very sensitive), consider feature selection to reduce multicollinearity
- For KNN: one-hot encode nominals, ordinal encode ordered, ALWAYS clip outliers (distance metric is sensitive)
""".strip()


def build_plan_prompt(
    dataset_name: str,
    columns: List[str],
    preview_rows: List[Dict[str, Any]],
    target_column: Optional[str] = None,
    dataset_profile: Optional[Dict[str, Any]] = None,
    aggressive_filter: bool = True,
) -> str:
    if aggressive_filter and dataset_profile and "columns" in dataset_profile:
        relevant_cols = []
        for c in dataset_profile["columns"]:
            name = c.get("name", "").lower()
            missing_ratio = c.get("missing_ratio", 0.0)
            is_id = "id" in name or "nbr" in name
            if is_id or missing_ratio > 0.0:
                relevant_cols.append({"name": name, "missing_ratio": missing_ratio, "is_id_candidate": is_id})
        filtered_profile = {
            "dataset_info": "Dataset " + dataset_name + ".",
            "note": "Showing ONLY IDs and columns with missing values. Clean columns are hidden.",
            "problematic_columns": relevant_cols
        }
    else:
        filtered_profile = dataset_profile

    context = {"dataset_name": dataset_name, "target_column": target_column, "filtered_profile": filtered_profile}

    schema_hint = {
        "reasoning_steps": [
            "Found 'encounter_id'. It is an ID. I will drop it.",
            "Found missing data < 2% in 'gender'. I will drop those rows.",
            "Found missing data > 2% in 'weight'. I will impute.",
            "Dataset is medical, I will use ordinal encoding."
        ],
        "dataset_summary": "Dataset contains noisy IDs and requires smart imputation.",
        "diagnostics": [],
        "prognostics": [],
        "actions": [
            {"action": "drop_column", "rationale": "Drop useless administrative IDs.", "target_columns": ["some_useless_id", "row_number_id"]},
            {"action": "handle_missing", "rationale": "Impute high missingness.", "target_columns": [], "params": {"strategy": "impute"}},
            {"action": "encode_categorical", "rationale": "Medical dataset, using ordinal to avoid dimensionality curse.", "target_columns": [], "params": {"method": "ordinal"}},
            {"action": "normalize_text", "rationale": "Standardize strings.", "target_columns": [], "params": {}}
        ],
        "assumptions": ["Assuming ordinal encoding is best for high cardinality."]
    }

    return (
        "Analyze the filtered profile below and generate an AutoML cleaning plan.\n"
        "Return ONLY raw JSON matching exactly this structure:\n"
        + json.dumps(schema_hint, indent=2) + "\n\n"
        "FILTERED DATASET PROFILE:\n"
        + json.dumps(context, indent=2) + "\n"
    )


def build_plan_prompt_c3(
    dataset_name: str,
    columns: List[str],
    preview_rows: List[Dict[str, Any]],
    target_column: Optional[str] = None,
    dataset_profile: Optional[Dict[str, Any]] = None,
    aggressive_filter: bool = True,
    user_context: Optional[Dict[str, Any]] = None,
) -> str:
    if user_context is None:
        user_context = {}

    if aggressive_filter and dataset_profile and "columns" in dataset_profile:
        relevant_cols = []
        for c in dataset_profile["columns"]:
            name = c.get("name", "").lower()
            missing_ratio = c.get("missing_ratio", 0.0)
            is_id = "id" in name or "nbr" in name
            if is_id or missing_ratio > 0.0:
                relevant_cols.append({"name": name, "missing_ratio": missing_ratio, "is_id_candidate": is_id})
        filtered_profile = {
            "dataset_info": "Dataset " + dataset_name + ".",
            "note": "Showing ONLY IDs and columns with missing values. Clean columns are hidden.",
            "problematic_columns": relevant_cols
        }
    else:
        filtered_profile = dataset_profile

    context = {"dataset_name": dataset_name, "target_column": target_column, "filtered_profile": filtered_profile}

    schema_hint = {
        "reasoning_steps": [
            "Read the user context to understand domain and downstream model.",
            "Found 'encounter_id'. It is an ID. I will drop it.",
            "Found missing data > 2% in 'weight'. I will impute.",
            "User says downstream model is tree-based, so ordinal encoding is safe.",
            "Column 'gender' has 2 categories and no natural order - one_hot is better."
        ],
        "dataset_summary": "Dataset requires imputation and model-appropriate encoding.",
        "diagnostics": [],
        "prognostics": [],
        "actions": [
            {"action": "drop_column", "rationale": "Drop useless administrative IDs.", "target_columns": ["encounter_id"], "params": {}},
            {"action": "handle_missing", "rationale": "Impute missing values for model compatibility.", "target_columns": [], "params": {"strategy": "impute"}},
            {"action": "encode_categorical", "rationale": "Chosen based on downstream model family and column cardinality.", "target_columns": [], "params": {"method": "one_hot"}},
            {"action": "normalize_text", "rationale": "Standardize strings.", "target_columns": [], "params": {}}
        ],
        "assumptions": []
    }

    user_block_parts = []
    if user_context.get("dataset_description"):
        user_block_parts.append("DATASET DESCRIPTION: " + user_context["dataset_description"])
    if user_context.get("downstream_model"):
        user_block_parts.append("DOWNSTREAM ML MODEL: " + user_context["downstream_model"])
    if user_context.get("must_do"):
        user_block_parts.append("MUST DO (user requirement): " + ", ".join(user_context["must_do"]))
    if user_context.get("must_not_do"):
        user_block_parts.append("MUST NOT DO (user requirement): " + ", ".join(user_context["must_not_do"]))
    if user_context.get("notes"):
        user_block_parts.append("ADDITIONAL NOTES: " + user_context["notes"])

    user_block = "\n".join(user_block_parts) if user_block_parts else "No user context provided."

    return (
        "Analyze the filtered profile below and generate an AutoML cleaning plan.\n"
        "Return ONLY raw JSON matching exactly this structure:\n"
        + json.dumps(schema_hint, indent=2) + "\n\n"
        "USER CONTEXT (read this FIRST, it affects your encoding and action choices):\n"
        + user_block + "\n\n"
        "FILTERED DATASET PROFILE:\n"
        + json.dumps(context, indent=2) + "\n"
    )


def build_plan_prompt_c4(
    dataset_name: str,
    columns: List[str],
    preview_rows: List[Dict[str, Any]],
    target_column: Optional[str] = None,
    dataset_profile: Optional[Dict[str, Any]] = None,
    aggressive_filter: bool = False,  # C4: show ALL columns, not just problematic
    user_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    C4 prompt: expanded action space, per-column encoding, full profile.
    Key difference from C3: shows ALL columns (not just missing/ID) so the LLM
    can make per-column encoding decisions and identify feature selection targets.
    """
    if user_context is None:
        user_context = {}

    # C4: Show the FULL profile so the LLM can reason about each column
    if dataset_profile and "columns" in dataset_profile:
        col_info = []
        for c in dataset_profile["columns"]:
            info = {
                "name": c.get("name", ""),
                "dtype": c.get("dtype", "unknown"),
                "missing_ratio": c.get("missing_ratio", 0.0),
                "n_unique": c.get("n_unique", None),
            }
            # Add value samples for categorical columns
            if c.get("dtype") in ("object", "string", "category"):
                info["sample_values"] = c.get("top_values", c.get("sample_values", []))[:5]
            # Flag ID candidates
            name_lower = c.get("name", "").lower()
            if "id" in name_lower or "nbr" in name_lower:
                info["is_id_candidate"] = True
            col_info.append(info)

        full_profile = {
            "dataset_info": f"Dataset '{dataset_name}' with {len(col_info)} columns.",
            "target_column": target_column,
            "columns": col_info,
        }
    else:
        full_profile = dataset_profile

    # Build the schema hint showing the EXPANDED action set
    schema_hint = {
        "reasoning_steps": [
            "Read user context: downstream model is Random Forest (tree-based).",
            "Column 'education' has natural order (basic < high_school < university) → ordinal.",
            "Column 'occupation' is nominal (14 categories, no order) → ordinal for tree model.",
            "Column 'race' is nominal (5 categories) → ordinal for tree model.",
            "Columns 'education' and 'education_num' encode same info → drop 'education_num'.",
            "Column 'fnlwgt' is census weight, not a predictor → drop it.",
            "Numeric columns have outliers → clip with IQR."
        ],
        "dataset_summary": "Census dataset requires per-column encoding, feature selection, and outlier treatment.",
        "diagnostics": [],
        "actions": [
            {
                "action": "handle_missing",
                "rationale": "Impute missing values.",
                "target_columns": [],
                "params": {"strategy": "impute"}
            },
            {
                "action": "drop_column",
                "rationale": "Drop redundant and non-predictive columns.",
                "target_columns": ["education_num", "fnlwgt"],
                "params": {}
            },
            {
                "action": "encode_categorical_per_column",
                "rationale": "Per-column encoding based on column semantics and downstream model.",
                "target_columns": [],
                "params": {
                    "column_encodings": {
                        "education": "ordinal",
                        "workclass": "ordinal",
                        "occupation": "ordinal",
                        "race": "ordinal",
                        "sex": "ordinal",
                        "marital_status": "ordinal"
                    },
                    "default_method": "ordinal"
                }
            },
            {
                "action": "clip_outliers",
                "rationale": "Clip extreme numeric values to reduce noise.",
                "target_columns": [],
                "params": {"method": "iqr", "iqr_k": 1.5}
            },
            {
                "action": "normalize_text",
                "rationale": "Standardize text before encoding.",
                "target_columns": [],
                "params": {}
            }
        ]
    }

    # Build user context block - richer than C3
    user_block_parts = []
    if user_context.get("dataset_description"):
        user_block_parts.append("DATASET DESCRIPTION: " + user_context["dataset_description"])
    if user_context.get("downstream_model"):
        user_block_parts.append("DOWNSTREAM ML MODEL: " + user_context["downstream_model"])
    if user_context.get("column_semantics"):
        # C4 addition: explicit column-level hints
        user_block_parts.append("COLUMN SEMANTICS (which columns are ordinal vs nominal):")
        for col, info in user_context["column_semantics"].items():
            user_block_parts.append(f"  - {col}: {info}")
    if user_context.get("must_do"):
        user_block_parts.append("MUST DO: " + ", ".join(user_context["must_do"]))
    if user_context.get("must_not_do"):
        user_block_parts.append("MUST NOT DO: " + ", ".join(user_context["must_not_do"]))
    if user_context.get("redundant_features"):
        user_block_parts.append("REDUNDANT FEATURES (consider dropping): " +
                              ", ".join(user_context["redundant_features"]))
    if user_context.get("notes"):
        user_block_parts.append("ADDITIONAL NOTES: " + user_context["notes"])

    user_block = "\n".join(user_block_parts) if user_block_parts else "No user context provided."

    return (
        "Analyze the FULL dataset profile below and generate an optimal cleaning plan.\n"
        "You MUST use 'encode_categorical_per_column' instead of 'encode_categorical'.\n"
        "Also consider 'select_features', 'clip_outliers', and 'bin_numeric' where appropriate.\n"
        "Return ONLY raw JSON matching this structure:\n"
        + json.dumps(schema_hint, indent=2) + "\n\n"
        "USER CONTEXT (read this FIRST):\n"
        + user_block + "\n\n"
        "FULL DATASET PROFILE:\n"
        + json.dumps(full_profile, indent=2) + "\n"
    )
