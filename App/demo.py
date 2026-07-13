from pathlib import Path
import json

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_FOLDER = PROJECT_ROOT / "demo_case"

ORIGINAL_FILE = DEMO_FOLDER / "original.csv"
CLEANED_FILE = DEMO_FOLDER / "cleaned.csv"
AUDIT_FILE = DEMO_FOLDER / "audit_log.json"
PLAN_FILE = DEMO_FOLDER / "ai_plan.json"


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Agentic Data Wrangling — Public Demo",
    page_icon="🧹",
    layout="wide",
)


# ---------------------------------------------------------
# File loading
# ---------------------------------------------------------

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


required_files = [
    ORIGINAL_FILE,
    CLEANED_FILE,
    AUDIT_FILE,
    PLAN_FILE,
]

missing_files = [path.name for path in required_files if not path.exists()]

if missing_files:
    st.error(
        "The demonstration files could not be found: "
        + ", ".join(missing_files)
    )
    st.stop()


original_df = load_csv(ORIGINAL_FILE)
cleaned_df = load_csv(CLEANED_FILE)
audit_log = load_json(AUDIT_FILE)
ai_plan = load_json(PLAN_FILE)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("Agentic Data Wrangling")

st.subheader("Public demonstration of a completed data-wrangling workflow")

st.info(
    """
    **Public demonstration environment**

    This version presents a predefined data-wrangling case previously
    processed using the local LLM-based system.

    No live LLM inference is performed in this online environment.
    The displayed AI plan, audit log and resulting dataset correspond
    to the selected demonstration scenario.
    """
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

st.header("Workflow summary")

column_1, column_2, column_3, column_4 = st.columns(4)

column_1.metric(
    "Original rows",
    f"{len(original_df):,}",
)

column_2.metric(
    "Cleaned rows",
    f"{len(cleaned_df):,}",
)

column_3.metric(
    "Original columns",
    len(original_df.columns),
)

column_4.metric(
    "Cleaned columns",
    len(cleaned_df.columns),
)


original_missing = int(original_df.isna().sum().sum())
cleaned_missing = int(cleaned_df.isna().sum().sum())

original_duplicates = int(original_df.duplicated().sum())
cleaned_duplicates = int(cleaned_df.duplicated().sum())

comparison_1, comparison_2 = st.columns(2)

with comparison_1:
    st.metric(
        "Missing values",
        cleaned_missing,
        delta=cleaned_missing - original_missing,
        delta_color="inverse",
        help=f"Original dataset: {original_missing}",
    )

with comparison_2:
    st.metric(
        "Duplicate rows",
        cleaned_duplicates,
        delta=cleaned_duplicates - original_duplicates,
        delta_color="inverse",
        help=f"Original dataset: {original_duplicates}",
    )


# ---------------------------------------------------------
# Main tabs
# ---------------------------------------------------------

tab_overview, tab_original, tab_cleaned, tab_plan, tab_audit, tab_downloads = st.tabs(
    [
        "Overview",
        "Original dataset",
        "Cleaned dataset",
        "AI plan",
        "Audit log",
        "Downloads",
    ]
)


with tab_overview:
    st.subheader("Before and after comparison")

    left, right = st.columns(2)

    with left:
        st.markdown("### Original dataset")
        st.dataframe(
            original_df.head(20),
            use_container_width=True,
        )

    with right:
        st.markdown("### Cleaned dataset")
        st.dataframe(
            cleaned_df.head(20),
            use_container_width=True,
        )

    st.subheader("Missing values by column")

    missing_comparison = pd.DataFrame(
        {
            "Original": original_df.isna().sum(),
            "Cleaned": cleaned_df.isna().sum(),
        }
    )

    missing_comparison = missing_comparison[
        (missing_comparison["Original"] > 0)
        | (missing_comparison["Cleaned"] > 0)
    ]

    if missing_comparison.empty:
        st.success("No missing values are present in either dataset.")
    else:
        st.dataframe(
            missing_comparison,
            use_container_width=True,
        )
        st.bar_chart(missing_comparison)


with tab_original:
    st.subheader("Original dataset")

    st.write(
        f"{len(original_df):,} rows and "
        f"{len(original_df.columns)} columns."
    )

    st.dataframe(
        original_df,
        use_container_width=True,
        height=500,
    )

    st.subheader("Original data types")

    original_types = pd.DataFrame(
        {
            "Column": original_df.columns,
            "Data type": original_df.dtypes.astype(str).values,
            "Missing values": original_df.isna().sum().values,
            "Unique values": original_df.nunique(dropna=False).values,
        }
    )

    st.dataframe(
        original_types,
        use_container_width=True,
    )


with tab_cleaned:
    st.subheader("Cleaned dataset")

    st.write(
        f"{len(cleaned_df):,} rows and "
        f"{len(cleaned_df.columns)} columns."
    )

    st.dataframe(
        cleaned_df,
        use_container_width=True,
        height=500,
    )

    st.subheader("Cleaned data types")

    cleaned_types = pd.DataFrame(
        {
            "Column": cleaned_df.columns,
            "Data type": cleaned_df.dtypes.astype(str).values,
            "Missing values": cleaned_df.isna().sum().values,
            "Unique values": cleaned_df.nunique(dropna=False).values,
        }
    )

    st.dataframe(
        cleaned_types,
        use_container_width=True,
    )


with tab_plan:
    st.subheader("Precomputed AI preparation plan")

    st.caption(
        "This plan was generated previously using the local LLM "
        "environment and is shown here for reproducibility."
    )

    st.json(ai_plan, expanded=True)


with tab_audit:
    st.subheader("Transformation audit log")

    st.caption(
        "This record documents the transformations performed during "
        "the demonstrated workflow."
    )

    st.json(audit_log, expanded=True)


with tab_downloads:
    st.subheader("Download demonstration artefacts")

    original_bytes = original_df.to_csv(index=False).encode("utf-8")
    cleaned_bytes = cleaned_df.to_csv(index=False).encode("utf-8")

    audit_bytes = json.dumps(
        audit_log,
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")

    plan_bytes = json.dumps(
        ai_plan,
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")

    button_1, button_2 = st.columns(2)

    with button_1:
        st.download_button(
            label="Download original CSV",
            data=original_bytes,
            file_name="original.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            label="Download audit log",
            data=audit_bytes,
            file_name="audit_log.json",
            mime="application/json",
            use_container_width=True,
        )

    with button_2:
        st.download_button(
            label="Download cleaned CSV",
            data=cleaned_bytes,
            file_name="cleaned.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            label="Download AI plan",
            data=plan_bytes,
            file_name="ai_plan.json",
            mime="application/json",
            use_container_width=True,
        )