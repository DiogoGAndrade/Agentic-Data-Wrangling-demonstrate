"""
Agentic Data Wrangling Streamlit UI
Mixed-initiative: LLM proposes, user approves, deterministic engine executes.

Public Streamlit deployment:
- loads the exported demonstration dataset by default;
- allows the user to upload a small CSV;
- calls a real external 3B model through the Hugging Face Inference API;
- executes approved actions through the original deterministic engine.
"""

import sys
import csv
import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ── Project paths ──────────────────────────────────────────────────────────────
# This must come before importing engine or llm modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Public deployment defaults ────────────────────────────────────────────────
DEMO_FOLDER = PROJECT_ROOT / "demo_case"
DEFAULT_DATASET_FILE = DEMO_FOLDER / "original.csv"

MAX_UPLOAD_MB = 10
MAX_UPLOAD_ROWS = 50_000
MAX_UPLOAD_COLUMNS = 100

PUBLIC_MODEL_LABEL = "Qwen 2.5 3B Instruct"


# ── Internal project imports ──────────────────────────────────────────────────
from engine.schemas import LLMPlan
from engine.actions import (
    fix_column_names,
    cast_type,
    handle_missing,
    normalize_text,
    deduplicate,
    drop_column,
    encode_categorical,
    remove_outliers,
)
from engine.provenance import ProvenanceLog, log_step
from engine.prognostics import compute_prognostics
from engine.profile_dataset import build_dataset_profile

from llm.prompt_templates import SYSTEM_PROMPT, build_plan_prompt
from llm.json_utils import extract_json


from llm.hf_api_client import HuggingFaceAPIClient

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Data Wrangling",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACTION_EXECUTORS = {
    "fix_column_names": fix_column_names,
    "cast_type": cast_type,
    "handle_missing": handle_missing,
    "normalize_text": normalize_text,
    "deduplicate": deduplicate,
    "drop_column": drop_column,
    "encode_categorical": encode_categorical,
    "remove_outliers": remove_outliers,
}

DEFAULT_PARAMS = {
    "fix_column_names": {},
    "cast_type": {"columns": ["col_name"], "dtype": "string", "errors": "raise"},
    "handle_missing": {"strategy": "impute", "columns": ["col_name"],
                       "impute": {"numeric": "median", "categorical": "most_frequent"}},
    "normalize_text": {"columns": ["col_name"], "ops": ["strip", "lower", "collapse_whitespace"]},
    "deduplicate": {"subset": None, "keep": "first", "case_insensitive": False},
    "drop_column": {"columns": ["col_name"], "reason": "user_request"},
    "encode_categorical": {"columns": ["col_name"], "method": "one_hot",
                           "drop_first": False, "max_categories": 30},
    "remove_outliers": {"columns": ["col_name"], "method": "iqr",
                        "iqr_k": 1.5, "mode": "drop_rows", "combine": "any"},
}

SEVERITY_ICON = {"critical": "●", "warning": "●", "info": "●"}
SEVERITY_COLOR = {"critical": "#c0392b", "warning": "#b45309", "info": "#1f3864"}

# Never push more than this many rows to the browser for a table preview.
# Large datasets (hundreds of thousands of rows) otherwise overflow Streamlit's
# websocket message limit and freeze the page. All cleaning still runs on the
# FULL dataset server-side; only the on-screen preview is bounded.
PREVIEW_ROWS = 500

# Actions whose engine executor requires params["columns"] as a non-empty list.
_COLUMN_ACTIONS = {"cast_type", "handle_missing", "normalize_text",
                   "drop_column", "encode_categorical", "remove_outliers"}


def resolve_action_columns(action_name, params, target_columns, df):
    """Bridge the plan schema (Action.target_columns) and the engine API
    (params['columns']).

    The LLM may put column names in the separate ``target_columns`` field,
    inside ``params['columns']``, or omit them entirely. The engine executors
    only read ``params['columns']``. Without this bridge, common actions such
    as normalize_text fail with "requires 'columns' as a non-empty list" even
    on valid plans. We:
      1. use params['columns'] if already present;
      2. else fall back to the action's target_columns;
      3. else, for text-only actions, default to all object/text columns so a
         column-less normalize_text still does something sensible.
    """
    params = dict(params) if params else {}
    if action_name not in _COLUMN_ACTIONS:
        return params
    cols = params.get("columns")
    if not cols:
        cols = list(target_columns) if target_columns else []
    # Sensible auto-defaults when the LLM names no columns: apply text/encoding
    # actions to all categorical columns rather than failing.
    if not cols and action_name in ("normalize_text", "encode_categorical"):
        cols = [c for c in df.columns if str(df[c].dtype) == "object"]
    if not cols and action_name in ("handle_missing", "remove_outliers"):
        # Default to the columns that actually have the relevant problem.
        if action_name == "handle_missing":
            cols = [c for c in df.columns if df[c].isnull().any()]
        else:
            cols = [c for c in df.columns
                    if str(df[c].dtype) != "object" and c not in (params.get("columns") or [])]
    if cols:
        params["columns"] = cols
    # Drop a target column that the LLM may have placed in target_columns but
    # which isn't present (test data), and never raise on a column the LLM
    # asked to drop that simply doesn't exist in this dataset.
    if action_name == "drop_column":
        params.setdefault("on_missing", "skip")
    return params

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Force light theme regardless of OS/browser dark mode ─────────── */
:root {
    --ink:        #1a2332;
    --ink-soft:   #4a5568;
    --line:       #e2e8f0;
    --surface:    #ffffff;
    --surface-alt:#f7f9fc;
    --accent:     #1f3864;
    --good:       #2e7d32;
    --warn:       #b45309;
}

/* ── Base: force white background and dark text everywhere ─────────── */
html, body { background-color: #f7f9fc !important; color: #1a2332 !important; }
.stApp { background-color: #f7f9fc !important; color: #1a2332 !important; }
.stApp * { color: inherit; }

/* ── Force all Streamlit text containers to be readable ────────────── */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
div[data-testid="stText"], div[data-testid="stCaptionContainer"],
label, .stSelectbox label, .stTextInput label, .stTextArea label,
.stFileUploader label, .stCheckbox label, .stRadio label,
div[class*="stMarkdown"] p, p, span:not([class*="badge"]) {
    color: #1a2332 !important;
}

/* ── Sidebar ────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0;
}
section[data-testid="stSidebar"] * { color: #1a2332 !important; }
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #1a2332 !important;
}

/* ── Headings ───────────────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    color: #1a2332 !important;
    font-weight: 600;
    letter-spacing: -0.01em;
    font-family: "Helvetica Neue", Arial, sans-serif;
}

/* ── Metric widgets ─────────────────────────────────────────────────── */
div[data-testid="stMetric"] { background: #ffffff; border-radius: 6px; padding: 8px 12px; }
div[data-testid="stMetricLabel"] > div { color: #4a5568 !important; font-size: 0.82rem; }
div[data-testid="stMetricValue"] > div { color: #1a2332 !important; font-weight: 700; }
div[data-testid="stMetricDelta"] svg { display: none; }

/* ── Tabs ───────────────────────────────────────────────────────────── */
div[data-testid="stTabs"] button { color: #1a2332 !important; }
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #1f3864 !important; border-bottom-color: #1f3864 !important;
}

/* ── Expander ───────────────────────────────────────────────────────── */
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary p,
div[data-testid="stExpander"] summary span { color: #1a2332 !important; }
div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 4px;
}

/* ── Inputs & selects ───────────────────────────────────────────────── */
input, textarea, select,
div[data-baseweb="select"] div,
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    color: #1a2332 !important;
    background-color: #ffffff !important;
}
div[data-baseweb="select"] [data-testid="stSelectboxVirtualDropdown"] {
    background-color: #ffffff !important;
    color: #1a2332 !important;
}
/* Visible resting border on inputs/selects/textareas so it is clear they are
   editable BEFORE the user clicks (previously the outline only appeared on focus). */
div[data-baseweb="select"] > div,
div[data-baseweb="input"],
div[data-baseweb="base-input"],
div[data-baseweb="textarea"],
.stTextInput div[data-baseweb="input"],
.stTextArea div[data-baseweb="textarea"],
.stSelectbox div[data-baseweb="select"] > div {
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 6px !important;
    background-color: #ffffff !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
/* Stronger highlight on hover/focus to signal interactivity */
div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"]:hover,
div[data-baseweb="textarea"]:hover {
    border-color: #94a3b8 !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"]:focus-within,
div[data-baseweb="textarea"]:focus-within {
    border-color: #1f3864 !important;
    box-shadow: 0 0 0 2px rgba(31,56,100,0.12) !important;
}
/* File uploader drop-zone: give it a clear dashed border too */
section[data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed #cbd5e1 !important;
    background-color: #ffffff !important;
    border-radius: 6px !important;
}
section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #1f3864 !important;
}

/* ── Alerts (info / warning / error / success) ──────────────────────── */
div[data-testid="stAlert"] { color: #1a2332 !important; }
div[data-testid="stAlert"] p { color: #1a2332 !important; }

/* ── Dataframe / table ──────────────────────────────────────────────── */
div[data-testid="stDataFrame"] iframe { background: #ffffff; }

/* ── Code blocks ────────────────────────────────────────────────────── */
div[data-testid="stCode"] { background: #f1f3f5 !important; }
div[data-testid="stCode"] code { color: #1a2332 !important; }

/* ── Buttons ────────────────────────────────────────────────────────── */
.stButton > button { color: #1a2332 !important; border: 1.5px solid #cbd5e1 !important;
    background-color: #ffffff !important; border-radius: 6px !important; }
.stButton > button:hover { border-color: #1f3864 !important; }
/* Primary buttons: navy background with FORCED white text on every child node.
   Streamlit nests the label in <p>/<div>/<span>, which otherwise inherit the
   global dark text rule and render the label dark-on-navy (illegible).
   NOTE: Streamlit >=1.40 dropped the button[kind="primary"] attribute and now
   marks primary buttons with data-testid="stBaseButton-primary". We target BOTH
   so the white label holds across Streamlit versions. */
.stButton > button[kind="primary"],
.stButton > button[kind="primary"] *,
.stButton > button[data-testid="stBaseButton-primary"],
.stButton > button[data-testid="stBaseButton-primary"] *,
button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primary"] *,
button[kind="primary"],
button[kind="primary"] * {
    background-color: #1f3864 !important;
    color: #ffffff !important;
    fill: #ffffff !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover *,
.stButton > button[data-testid="stBaseButton-primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover *,
button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover * {
    background-color: #16294a !important;
    color: #ffffff !important;
    fill: #ffffff !important;
}

/* ── Download buttons ───────────────────────────────────────────────── */
.stDownloadButton > button { color: #1a2332 !important;
    border: 1.5px solid #cbd5e1 !important; border-radius: 6px !important; }
.stDownloadButton > button:hover { border-color: #1f3864 !important; }
.stDownloadButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] *,
.stDownloadButton > button[data-testid="stBaseButton-primary"],
.stDownloadButton > button[data-testid="stBaseButton-primary"] * { color: #ffffff !important; fill: #ffffff !important; }

/* ── Progress bar ───────────────────────────────────────────────────── */
div[data-testid="stProgressBar"] > div { background-color: #1f3864 !important; }

/* ── Font ───────────────────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: "Helvetica Neue", Arial, sans-serif; }

/* ── Custom components ──────────────────────────────────────────────── */
.app-header { border-bottom: 2px solid #1f3864; padding-bottom: 14px; margin-bottom: 6px; }
.app-title  { font-size: 1.7rem; font-weight: 700; color: #1f3864 !important; margin: 0; }
.app-sub    { font-size: 0.95rem; color: #4a5568 !important; margin-top: 4px; }

/* ── Stepper: top-of-page progress across the 4 phases ──────────────── */
.stepper { display: flex; align-items: flex-start; justify-content: space-between;
    margin: 6px 0 4px 0; padding: 0; }
.stepper-item { display: flex; flex-direction: column; align-items: center;
    flex: 1 1 0; position: relative; text-align: center; }
/* connector line between circles */
.stepper-item:not(:last-child)::after {
    content: ""; position: absolute; top: 16px; left: 50%; width: 100%;
    height: 3px; background: #d6e0f0; z-index: 0; }
.stepper-item.done:not(:last-child)::after { background: #2e7d32; }
.stepper-circle {
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 15px; z-index: 1; position: relative;
    border: 3px solid #d6e0f0; background: #ffffff; color: #94a3b8; }
.stepper-item.done .stepper-circle  { background:#2e7d32; border-color:#2e7d32; color:#ffffff; }
.stepper-item.active .stepper-circle{ background:#1f3864; border-color:#1f3864; color:#ffffff;
    box-shadow: 0 0 0 4px rgba(31,56,100,0.15); }
.stepper-label { margin-top: 8px; font-size: 0.82rem; font-weight: 600;
    color: #94a3b8; line-height: 1.2; max-width: 130px; }
.stepper-item.done .stepper-label   { color: #2e7d32; }
.stepper-item.active .stepper-label { color: #1f3864; }
.stepper-sub { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.05em;
    text-transform: uppercase; color: #cbd5e1; margin-bottom: 2px; }
.stepper-item.done .stepper-sub   { color: #2e7d32; }
.stepper-item.active .stepper-sub { color: #1f3864; }

/* Phase header bar: a clear banner that separates each of the 4 phases.
   Navy left rule + tinted background make the boundary obvious. */
.step-header {
    display: flex; align-items: center;
    background: #eef2f9;
    border: 1px solid #d6e0f0;
    border-left: 5px solid #1f3864;
    border-radius: 8px;
    padding: 12px 16px; margin: 8px 0 16px 0;
}
.step-header .step-badge,
.step-badge {
    display: inline-flex; align-items: center; justify-content: center;
    background: #1f3864 !important; color: #ffffff !important;
    border-radius: 50%; width: 28px; height: 28px; min-width: 28px;
    text-align: center; line-height: 28px;
    font-weight: 700; font-size: 14px; margin-right: 12px; flex: 0 0 auto;
}
/* Force the number itself white: the global dark-text rule otherwise wins
   on the text node inside the badge span. */
.step-header .step-badge, .step-badge,
.step-header .step-badge *, .step-badge * { color: #ffffff !important; }
.step-title { font-size: 1.2rem; font-weight: 700; color: #1f3864 !important; }
.action-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-left: 3px solid #1f3864;
    border-radius: 6px; padding: 12px 16px; margin-bottom: 8px;
    color: #1a2332 !important;
}
.action-card strong, .action-card span { color: #1a2332 !important; }
.badge-win  { background:#e8f3ec; color:#2e7d32 !important; border:1px solid #bfe0c8; padding:2px 9px; border-radius:4px; font-size:12px; font-weight:600; }
.badge-info { background:#eaf0f8; color:#1f3864 !important; border:1px solid #c5d6ee; padding:2px 9px; border-radius:4px; font-size:12px; font-weight:600; }
hr.divider  { border: none; border-top: 2px solid #d6e0f0; margin: 30px 0; }
.section-label { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; color: #4a5568 !important; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Auto-detect helpers ───────────────────────────────────────────────────────
def _sniff_sep(file) -> str:
    try:
        raw = file.read(2048).decode("utf-8", errors="ignore")
        file.seek(0)
        dialect = csv.Sniffer().sniff(raw, delimiters=",;\t|")
        return dialect.delimiter
    except Exception:
        return ","

def _sniff_encoding(file) -> str:
    try:
        import chardet
        raw = file.read(4096); file.seek(0)
        result = chardet.detect(raw)
        enc = result.get("encoding") or "utf-8"
        return enc if enc.lower() != "ascii" else "utf-8"
    except ImportError:
        return "utf-8"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">System</div>', unsafe_allow_html=True)
    st.markdown("### Agentic Data Wrangling")
    st.markdown("*LLM proposes · You approve · Engine executes*")
    st.markdown("---")

    # 1. Upload
    st.markdown("#### Your dataset")
    dataset_name = st.text_input("Dataset name", "My Dataset",
                                  help="A friendly label, used in exported filenames.")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"],
                                 help="Drag & drop or browse. Max 10 MB.")

    # 2. File format
    st.markdown("#### File format")
    st.caption("Not sure? Leave everything on **Auto-detect**. It works for most files.")

    sep_options = {
        "Auto-detect": "__auto__",
        "Comma  ( , )": ",",
        "Semicolon  ( ; )": ";",
        "Tab": "\t",
        "Pipe  ( | )": "|",
    }
    sep_label = st.selectbox(
        "Column separator",
        list(sep_options.keys()), index=0, key="set_sep",
        help="The character that separates columns in your CSV.")
    sep = sep_options[sep_label]

    enc_options = {
        "Auto-detect": "__auto__",
        "UTF-8 (most common)": "utf-8",
        "Latin-1 / Excel Western Europe": "latin-1",
        "Windows-1252": "cp1252",
        "UTF-16": "utf-16",
    }
    enc_label = st.selectbox(
        "Text encoding",
        list(enc_options.keys()), index=0, key="set_enc",
        help="How characters are stored in the file. UTF-8 works for most modern files. "
             "If you see garbled text like 'Ã©' instead of 'é', try Latin-1.")
    enc_choice = enc_options[enc_label]

    na_options = {
        "Standard (NA, N/A, null, NULL, empty)": "NA,N/A,null,NULL,",
        "Only empty cells": "",
        "Custom…": "__custom__",
    }
    na_label = st.selectbox(
        "Missing value markers",
        list(na_options.keys()), index=0, key="set_na",
        help="Values in your file that represent missing data and should be treated as empty.")
    if na_label == "Custom…":
        na_raw = st.text_input("Enter markers (comma-separated)", "NA,N/A,?,missing")
    else:
        na_raw = na_options[na_label]
    na_values = [x.strip() for x in na_raw.split(",") if x.strip()]

    if uploaded is not None:
        st.caption("File uploaded. Format will be auto-detected if set to Auto.")

    st.markdown("---")

    # 3. AI model
    st.markdown("#### AI model")
    st.caption("Runs entirely on your computer. No data leaves your machine.")

    st.success(f"Online model: {PUBLIC_MODEL_LABEL}")
    st.caption(
        "The model runs through the Hugging Face Inference API. "
        "The first request or a busy provider may take longer."
    )
    model = PUBLIC_MODEL_LABEL

    st.markdown("---")

    # 4. Target + downstream model
    st.markdown("#### About your task")
    _target_placeholder = "(none)"

# ── Load dataset ──────────────────────────────────────────────────────────────
toy_rows = [
    {"age": 39, "workclass": " State-gov",       "education": "Bachelors", "hours_per_week": 40, "income": "<=50K"},
    {"age": 50, "workclass": " Self-emp-not-inc", "education": "Bachelors", "hours_per_week": 13, "income": "<=50K"},
    {"age": None,"workclass": " Private",         "education": "HS-grad",   "hours_per_week": 40, "income": "<=50K"},
    {"age": 53, "workclass": " Private",          "education": "11th",      "hours_per_week": 40, "income": "<=50K"},
    {"age": 28, "workclass": " Private",          "education": "Bachelors", "hours_per_week": 40, "income": ">50K"},
    {"age": 37, "workclass": " Private",          "education": "Masters",   "hours_per_week": 60, "income": ">50K"},
]
df_fallback = pd.DataFrame(toy_rows)

def _load_csv(file, sep, enc_choice, na_values):
    actual_sep = _sniff_sep(file) if sep == "__auto__" else sep
    actual_enc = _sniff_encoding(file) if enc_choice == "__auto__" else enc_choice
    return pd.read_csv(file, sep=actual_sep, encoding=actual_enc,
                       na_values=na_values, keep_default_na=True, low_memory=False)

if uploaded is not None:
    try:
        file_size_mb = uploaded.size / (1024 * 1024)
        if file_size_mb > MAX_UPLOAD_MB:
            st.sidebar.error(
                f"The public demonstration accepts CSV files up to {MAX_UPLOAD_MB} MB."
            )
            st.stop()

        df0 = _load_csv(
            uploaded,
            sep=sep,
            enc_choice=enc_choice,
            na_values=na_values,
        )

        if len(df0.columns) > MAX_UPLOAD_COLUMNS:
            st.sidebar.error(
                f"The uploaded dataset has {len(df0.columns)} columns. "
                f"The public limit is {MAX_UPLOAD_COLUMNS}."
            )
            st.stop()

        if len(df0) > MAX_UPLOAD_ROWS:
            st.sidebar.warning(
                f"The uploaded dataset has {len(df0):,} rows. "
                f"For the public demonstration, only the first "
                f"{MAX_UPLOAD_ROWS:,} rows will be used."
            )
            df0 = df0.head(MAX_UPLOAD_ROWS).copy()

    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")
        st.stop()

elif DEFAULT_DATASET_FILE.exists():
    try:
        df0 = pd.read_csv(DEFAULT_DATASET_FILE, low_memory=False)
        dataset_name = "Diabetes"
        st.sidebar.info(
            "The demonstration dataset is loaded. "
            "Upload a CSV above to replace it."
        )
    except Exception as e:
        st.error(f"Could not load the demonstration dataset: {e}")
        st.stop()
else:
    df0 = df_fallback.copy()
    st.sidebar.warning(
        "The file demo_case/original.csv was not found, "
        "so the small built-in fallback dataset is being used."
    )


# Session state init
_defaults = {
    "df_original": df0.copy(),
    "df_current": df0.copy(),
    "provenance": ProvenanceLog(),
    "last_plan": None,
    "uploaded_name": getattr(uploaded, "name", "__default_demo__"),
    "step": 1,
    "execution_done": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

uploaded_name = getattr(uploaded, "name", "__default_demo__")
if uploaded_name != st.session_state.uploaded_name:
    st.session_state.uploaded_name = uploaded_name
    st.session_state.df_original = df0.copy()
    st.session_state.df_current = df0.copy()
    st.session_state.provenance = ProvenanceLog()
    st.session_state.last_plan = None
    st.session_state.execution_done = False
    st.session_state.step = 1
    st.sidebar.success("Dataset loaded.")

# ── Sidebar continued: target + downstream model ──────────────────────────────
cols = list(st.session_state.df_current.columns)
with st.sidebar:
    default_idx = 0
    for preferred in ["target", "income", "label", "class", "y"]:
        if preferred in cols:
            default_idx = cols.index(preferred)
            break
    # Does the dataset already contain a plausible target column?
    _has_target_candidate = any(
        c.lower() in {"target", "income", "label", "class", "y", "outcome",
                      "price", "saleprice", "diagnosis"}
        for c in cols)

    data_role = st.radio(
        "Is this training data or test/inference data?",
        ["Auto-detect", "Training data (target column is present)",
         "Test / inference data (no target column)"],
        index=0, key="set_data_role",
        help="Training data contains the column you want to predict. "
             "Test/inference data does not: it is the data you will run the "
             "trained model on. The system cleans them differently and must "
             "never invent or impute the target.")

    if data_role == "Auto-detect":
        is_test = not _has_target_candidate
        if is_test:
            st.caption("No obvious target column found → treating this as "
                       "**test / inference data**. Confirm below.")
        else:
            st.caption("A target-like column was found → treating this as "
                       "**training data**.")
    else:
        is_test = data_role.startswith("Test")

    target_type = None
    if not is_test:
        # Training data: pick the target column from the dataset.
        target_column = st.selectbox(
            "What are you trying to predict?",
            options=[_target_placeholder] + cols,
            index=1 + default_idx if cols else 0,
            help="The column your ML model will predict. "
                 "The system will make sure it is never accidentally dropped "
                 "or corrupted.")
        target_column = None if target_column == _target_placeholder else target_column
    else:
        # Test data: the target is absent, so the user must DECLARE it
        # (name + type) so the engine knows what is being predicted and can
        # protect/align columns without fabricating the label.
        st.markdown("**Declare the target (required for test data)**")
        target_column = st.text_input(
            "Target variable name",
            value="",
            placeholder="e.g. income, SalePrice, diagnosis",
            help="The name of the column your trained model predicts. "
                 "It is not in this file, but the system needs it to keep the "
                 "feature columns consistent with your training data.")
        target_type = st.selectbox(
            "Target type",
            ["Classification (categories)", "Regression (numeric)"],
            index=0,
            help="Whether the model predicts a category or a number. "
                 "This guides safe handling of the remaining columns.")
        target_column = target_column.strip() or None
        if target_column is None:
            st.warning("Enter the target variable name to continue with test data.")

    # Map the user's model family to a preprocessing PROFILE. The key question
    # for cleaning is: does the model need feature scaling / careful imputation
    # (linear & distance-based) or is it scale-invariant (trees)?
    #   - When unsure or training several models, default to the SAFER profile
    #     (scaling-sensitive), because under-cleaning a tree model is harmless
    #     but under-cleaning a linear/KNN/NN model degrades it.
    _model_labels = {
        "I don't know yet (use safe defaults)":            "ScalingSensitive",
        "Tree-based (Random Forest, Gradient Boosting, XGBoost)": "TreeBased",
        "Linear (Logistic Regression, Ridge, Linear/SVM)": "ScalingSensitive",
        "K-Nearest Neighbours (KNN)":                      "ScalingSensitive",
        "Neural Network / Deep Learning":                  "ScalingSensitive",
        "Several different models":                        "ScalingSensitive",
    }
    downstream_label = st.selectbox(
        "Which ML model will you train?",
        list(_model_labels.keys()), index=0, key="set_downstream",
        help="This adjusts the cleaning strategy. Tree models are scale-"
             "invariant; linear, KNN and neural models need feature scaling and "
             "careful imputation. If unsure or training several models, the safe "
             "default assumes scaling is needed.")
    downstream_model = _model_labels[downstream_label]

    extra_context = st.text_area(
        "Additional information for the AI (optional)",
        placeholder=(
            "Anything the AI should know about your model or data. Examples:\n"
            "• 'This is medical data; missing values are often meaningful.'\n"
            "• 'I will train an XGBoost model with class weights.'\n"
            "• 'Do not treat 0 in the glucose column as a real value.'"),
        height=110,
        help="Free-text context about your dataset or downstream model. "
             "It is added to the AI prompt to produce a more tailored cleaning plan. "
             "Not required; leave blank to use automatic behaviour.")

    st.markdown("---")
    if st.button("Reset settings", use_container_width=True,
                 help="Reset only the sidebar settings (file format, AI model, "
                      "target, downstream model) back to their defaults. "
                      "Does not touch your data or cleaning steps."):
        for _k in ("set_sep", "set_enc", "set_na", "set_llm", "set_data_role",
                   "set_downstream", "set_ollama_url", "set_custom_tag"):
            st.session_state.pop(_k, None)
        st.rerun()

    if st.button("Reset everything", use_container_width=True,
                 help="Undo all cleaning steps, clear the AI plan, and start fresh "
                      "from your original uploaded data."):
        st.session_state.df_current     = st.session_state.df_original.copy()
        st.session_state.provenance     = ProvenanceLog()
        st.session_state.last_plan      = None
        st.session_state.execution_done = False
        st.session_state.step           = 1
        # Clear transient widget state so a fresh plan starts clean:
        # per-action approval checkboxes (approve_0, approve_1, …) and the
        # manual-override scratch keys. Without this, stale checkbox state
        # leaks into the next plan and the reset feels like it "did nothing".
        for _k in [k for k in st.session_state.keys()
                   if k.startswith("approve_")]:
            del st.session_state[_k]
        for _k in ("_manual_last", "_manual_params", "manual_params_text",
                   "manual_action_select"):
            st.session_state.pop(_k, None)
        st.rerun()

# ── Main header ───────────────────────────────────────────────────────────────
st.markdown('<div class="app-header"><div class="app-title">Agentic Data Wrangling</div><div class="app-sub">An auditable, LLM-powered assistant for model-aware data preparation</div></div>', unsafe_allow_html=True)
st.markdown(
    "**Model-aware, mixed-initiative data preparation.** "
    "Upload your dataset → the AI proposes a cleaning plan → "
    "you inspect and approve each step → the engine executes it deterministically "
    "with a full audit trail.")


# ── Progress bar ──────────────────────────────────────────────────────────────
STEPS = ["Upload & Preview", "AI Plan", "Review & Execute", "Export"]
_n = st.session_state.step
_items = ""
for i, label in enumerate(STEPS):
    if i + 1 < _n:
        cls, mark = "done", "✓"
    elif i + 1 == _n:
        cls, mark = "active", str(i + 1)
    else:
        cls, mark = "", str(i + 1)
    sub = "Done" if cls == "done" else ("Current" if cls == "active" else f"Step {i+1}")
    _items += (
        f"<div class='stepper-item {cls}'>"
        f"<div class='stepper-circle'>{mark}</div>"
        f"<div class='stepper-sub'>{sub}</div>"
        f"<div class='stepper-label'>{label}</div>"
        f"</div>")
st.markdown(f"<div class='stepper'>{_items}</div>", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 - UPLOAD & PREVIEW
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="step-header"><span class="step-badge">1</span>'
            '<span class="step-title">Upload &amp; Preview</span></div>',
            unsafe_allow_html=True)

df = st.session_state.df_current
n_missing = int(df.isnull().sum().sum())
n_dupes   = int(df.duplicated().sum())
n_cat     = int((df.dtypes == "object").sum())

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Rows", f"{len(df):,}")
m2.metric("Columns", len(df.columns))
m3.metric("Missing values", n_missing,
          delta=f"{100*n_missing/max(len(df)*len(df.columns),1):.1f}%" if n_missing else None,
          delta_color="inverse")
m4.metric("Duplicate rows", n_dupes, delta_color="inverse")
m5.metric("Categorical columns", n_cat)

tab_data, tab_types, tab_stats = st.tabs(["Data", "Column types & missing", "Statistics"])
with tab_data:
    if len(df) > PREVIEW_ROWS:
        st.caption(f"Showing the first {PREVIEW_ROWS:,} of {len(df):,} rows "
                   f"(preview only; all {len(df):,} rows are cleaned).")
    st.dataframe(df.head(PREVIEW_ROWS), use_container_width=True, height=280)
with tab_types:
    type_df = pd.DataFrame({
        "Column":   df.columns,
        "Type":     [str(t) for t in df.dtypes],
        "Missing":  df.isnull().sum().values,
        "Missing%": [f"{100*v/max(len(df),1):.1f}%" for v in df.isnull().sum().values],
        "Unique values": df.nunique().values,
    })
    st.dataframe(type_df, use_container_width=True, hide_index=True)
with tab_stats:
    try:
        st.dataframe(df.describe(include="all").T, use_container_width=True)
    except Exception:
        st.caption("Statistics not available for this dataset.")

if st.session_state.step == 1:
    if st.button("→ Continue to AI Plan", type="primary"):
        st.session_state.step = 2
        st.rerun()

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 - LLM PLAN
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="step-header"><span class="step-badge">2</span>'
            '<span class="step-title">AI Plan Generation</span></div>',
            unsafe_allow_html=True)

if st.session_state.step < 2:
    st.info("Complete Step 1 first.")
else:
    # Prognostics - always shown, deterministic, no LLM needed
    with st.expander("Automatic data quality checks (no AI needed)", expanded=True):
        st.caption(
            "These checks run instantly on your data. "
            "They detect missing values, potential data leakage, "
            "redundant columns, and class imbalance. "
            "Results are included in the AI prompt automatically.")
        prog = compute_prognostics(st.session_state.df_current, target_column)
        if not prog:
            st.success("No issues detected.")
        else:
            for p in prog:
                sev  = p.get("severity", "info")
                icon = SEVERITY_ICON.get(sev, "●")
                msg  = f"{icon} **[{p['code']}]** {p['message']}"
                if sev == "critical":   st.error(msg)
                elif sev == "warning":  st.warning(msg)
                else:                   st.info(msg)
                with st.expander(f"Details: {p['code']}"):
                    st.json(p.get("evidence", {}))

    st.markdown("#### Ask the AI for a cleaning plan")
    st.caption("The AI reads the checks above and proposes a list of cleaning "
               "steps. Nothing is changed yet: you review and approve in Step 3.")
    col_inst, col_btn = st.columns([3, 1])
    with col_inst:
        custom_instruction = st.text_area(
            "Optional: give the AI a specific instruction (leave blank for automatic)",
            placeholder=(
                "Examples:\n"
                "• 'Do not drop any rows'\n"
                "• 'Focus on fixing missing values in the age column'\n"
                "• 'The workclass column has leading spaces, clean those'"),
            height=100,
        )
    with col_btn:
        st.markdown("<br><br>", unsafe_allow_html=True)
        gen_btn = st.button(
            "Propose plan (review first)",
            type="primary",
            use_container_width=True,
            help="Ask the 3B model to draft a cleaning plan. You approve each step in Step 3."
        )
        auto_btn = st.button(
            "Auto-clean (full autonomy)",
            use_container_width=True,
            help="The 3B model proposes a plan and applies every supported step automatically."
        )

    if gen_btn or auto_btn:
        try:
            client = HuggingFaceAPIClient.from_streamlit_secrets()
        except Exception as e:
            st.error(str(e))
            st.stop()
        # Give the planner a full statistical profile (missingness, cardinality,
        # ID-likeness, etc.), not just head(8). This matches what the thesis
        # evaluation pipeline sends, so the app produces comparable plans.
        _profile = build_dataset_profile(
            st.session_state.df_current, target_column=target_column)
        prompt = build_plan_prompt(
            dataset_name=dataset_name,
            columns=list(st.session_state.df_current.columns),
            preview_rows=st.session_state.df_current.head(8).to_dict(orient="records"),
            target_column=target_column,
            dataset_profile=_profile,
        )
        # Make the plan MODEL-AWARE: tell the LLM the downstream model's
        # preprocessing profile so encoding/scaling/imputation choices match it.
        if downstream_model == "TreeBased":
            _model_hint = ("The downstream model is TREE-BASED (e.g. Random Forest, "
                           "Gradient Boosting, XGBoost): it is scale-invariant, so do "
                           "NOT add feature scaling; ordinal encoding of high-cardinality "
                           "categoricals is acceptable.")
        else:
            _model_hint = ("The downstream model is SCALING-SENSITIVE (linear, KNN, "
                           "SVM or neural network): features MUST be scaled, missing "
                           "values imputed carefully, and one-hot encoding is preferred "
                           "for low-cardinality nominal categoricals.")
        prompt = (f"DOWNSTREAM MODEL: {_model_hint}\n\n{prompt}")

        if custom_instruction.strip():
            prompt = f"Additional instruction from user: {custom_instruction.strip()}\n\n{prompt}"
        if extra_context.strip():
            prompt = (f"Additional context about the dataset and downstream model "
                      f"(provided by the user): {extra_context.strip()}\n\n{prompt}")
        if is_test:
            _ttype = (target_type or "unknown").split(" (")[0].lower()
            prompt = (f"IMPORTANT: This is TEST / INFERENCE data. The target column "
                      f"'{target_column}' is NOT present and must NOT be created, "
                      f"imputed, or inferred. The prediction task is {_ttype}. "
                      f"Clean only the feature columns and keep them aligned with the "
                      f"training schema.\n\n{prompt}")

        with st.spinner(f"Asking {model} for a cleaning plan… (this may take 10–60 seconds)"):
            raw = client.generate(prompt=prompt, system=SYSTEM_PROMPT, temperature=0.2)

        with st.expander("Raw AI output (for debugging)"):
            st.code(str(raw), language="json")

        try:
            obj  = extract_json(raw)
            plan = LLMPlan.model_validate(obj)
            st.session_state.last_plan = plan
            st.session_state.step = 3
            # Full-autonomy path: flag the plan to run automatically in Step 3.
            st.session_state.auto_run = bool(auto_btn)
            st.success(f"Plan received: {len(plan.actions)} cleaning action(s) proposed.")
            st.rerun()
        except Exception as e:
            st.error(f"Could not parse the AI response: {e}")
            st.caption("Try regenerating, or switch to a larger model in the sidebar.")

    if st.session_state.last_plan and st.session_state.step >= 3:
        n = len(st.session_state.last_plan.actions)
        st.success(f"Plan ready: {n} action(s) proposed. Scroll down to Step 3 to review.")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 - REVIEW & EXECUTE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="step-header"><span class="step-badge">3</span>'
            '<span class="step-title">Review &amp; Execute</span></div>',
            unsafe_allow_html=True)

plan: LLMPlan | None = st.session_state.last_plan

if st.session_state.step < 3 or plan is None:
    st.info("Generate a plan in Step 2 first.")
else:
    # LLM diagnostics (if any)
    if plan.diagnostics:
        with st.expander("AI diagnostics (issues the AI flagged)", expanded=False):
            for d in plan.diagnostics:
                sev  = d.severity.value
                icon = SEVERITY_ICON.get(sev, "●")
                msg  = f"{icon} [{d.code}] {d.message}"
                if sev == "critical":   st.error(msg)
                elif sev == "warning":  st.warning(msg)
                else:                   st.info(msg)

    st.markdown(
        f"The AI proposed **{len(plan.actions)} action(s)**. "
        "Tick the ones you want to apply. Untick to skip any you disagree with. "
        "Then click **Execute**.")

    approved = []
    for i, a in enumerate(plan.actions):
        col_chk, col_info = st.columns([0.06, 0.94])
        with col_chk:
            is_ok = st.checkbox(
                f"Approve action {i+1}: {a.action.value}",
                value=True, key=f"approve_{i}",
                label_visibility="collapsed")
        with col_info:
            badge = "badge-win" if is_ok else "badge-info"
            status_text = "✓ Will apply" if is_ok else "Skip"
            st.markdown(
                f"<div class='action-card'>"
                f"<span class='{badge}'>{status_text}</span>&nbsp;&nbsp;"
                f"<strong>{a.action.value.replace('_', ' ').title()}</strong>"
                f": {a.rationale}"
                f"</div>",
                unsafe_allow_html=True)
            if a.params:
                with st.expander(f"Parameters for {a.action.value}"):
                    st.json(a.params)
        if is_ok:
            approved.append(a)

    st.markdown(f"**{len(approved)} of {len(plan.actions)} actions selected.**")

    col_run, col_rst = st.columns(2)
    with col_run:
        run_btn = st.button(
            f"▶ Execute {len(approved)} action(s)", type="primary",
            use_container_width=True, disabled=(len(approved) == 0))
    with col_rst:
        if st.button("↩ Undo all (reset to original)", use_container_width=True):
            st.session_state.df_current     = st.session_state.df_original.copy()
            st.session_state.provenance     = ProvenanceLog()
            st.session_state.execution_done = False
            st.rerun()

    # Full-autonomy: if the user clicked "Auto-clean everything", run all actions
    # automatically this once, without waiting for the Execute click.
    _auto = st.session_state.pop("auto_run", False)
    if _auto:
        approved = list(plan.actions)
        st.info("Auto-clean: applying all proposed actions automatically.")

    # Order actions by a sane wrangling sequence so that, e.g., text is
    # normalised BEFORE it is encoded (otherwise normalize_text finds no text
    # columns left). Ties keep the LLM's original order (stable sort).
    _ORDER = {"fix_column_names": 0, "normalize_text": 1, "cast_type": 2,
              "deduplicate": 3, "handle_missing": 4, "remove_outliers": 5,
              "drop_column": 6, "encode_categorical": 7}
    approved = sorted(approved, key=lambda a: _ORDER.get(a.action.value, 9))

    if (run_btn or _auto) and approved:
        df_before = st.session_state.df_current.copy()
        results   = []
        progress  = st.progress(0, text="Starting…")

        for step_idx, a in enumerate(approved):
            progress.progress(step_idx / len(approved),
                              text=f"Applying: {a.action.value.replace('_', ' ')}…")
            df_cur        = st.session_state.df_current
            before_schema = {c: str(df_cur[c].dtype) for c in df_cur.columns}
            executor      = ACTION_EXECUTORS.get(a.action.value)

            if executor is None:
                results.append({"Action": a.action.value, "Result": "Not implemented",
                                 "Detail": "Skipped"})
                continue

            try:
                exec_params = resolve_action_columns(
                    a.action.value, a.params,
                    getattr(a, "target_columns", None), df_cur)
                if a.action.value == "drop_column":
                    exec_params.setdefault("target_column", target_column)

                # Graceful no-op: a column action with no applicable columns
                # (e.g. normalize_text after everything was already encoded)
                # should be skipped, not crashed.
                if (a.action.value in _COLUMN_ACTIONS
                        and not exec_params.get("columns")):
                    results.append({"Action": a.action.value.replace("_", " ").title(),
                                     "Result": "Skipped",
                                     "Detail": "No applicable columns"})
                    continue

                df_after, diff, warnings = executor(df_cur, exec_params)
                after_schema = {c: str(df_after[c].dtype) for c in df_after.columns}
                st.session_state.df_current = df_after

                st.session_state.provenance.add_step(log_step(
                    dataset_id=dataset_name, step_id=step_idx,
                    action_name=a.action.value, approved=True, status="applied",
                    params=exec_params, rationale=a.rationale,
                    before_schema=before_schema, after_schema=after_schema,
                    diff_summary=diff, warnings=warnings, error=None,
                ))
                rows_rm = (diff or {}).get("rows_removed", 0) or 0
                results.append({"Action": a.action.value.replace("_", " ").title(),
                                 "Result": "Applied",
                                 "Detail": f"{rows_rm} rows removed" if rows_rm else "OK"})
            except Exception as e:
                results.append({"Action": a.action.value, "Result": "Failed",
                                 "Detail": str(e)})

        progress.progress(1.0, text="Done.")
        st.session_state.execution_done = True
        st.session_state.step = 4

        st.success("Execution complete.")
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

        tab_before, tab_after, tab_diff = st.tabs(
            ["Dataset before", "Dataset after", "What changed"])
        with tab_before:
            st.dataframe(df_before.head(PREVIEW_ROWS), use_container_width=True, height=200)
        with tab_after:
            st.dataframe(st.session_state.df_current.head(PREVIEW_ROWS),
                         use_container_width=True, height=200)
        with tab_diff:
            rb, ra = len(df_before), len(st.session_state.df_current)
            cb, ca = set(df_before.columns), set(st.session_state.df_current.columns)
            st.markdown(f"- **Rows:** {rb:,} → {ra:,}  "
                        f"({'−' if ra <= rb else '+'}{abs(ra-rb)})")
            added   = ca - cb
            removed = cb - ca
            if added:   st.markdown(f"- **Columns added:** {', '.join(f'`{c}`' for c in added)}")
            if removed: st.markdown(f"- **Columns removed:** {', '.join(f'`{c}`' for c in removed)}")
            if not added and not removed:
                st.markdown("- **Columns:** unchanged")

    # ── Manual override (advanced, hidden by default) ──────────────────────────
    with st.expander("Apply a single manual action (advanced, no AI)"):
        st.caption(
            "Use this to apply one specific transformation directly, "
            "bypassing the AI. Useful for debugging or fine-tuning after the main plan.")
        manual_action = st.selectbox("Action", list(ACTION_EXECUTORS.keys()),
                                      key="manual_action_select")
        if st.session_state.get("_manual_last") != manual_action:
            st.session_state["_manual_last"]   = manual_action
            st.session_state["_manual_params"] = DEFAULT_PARAMS.get(manual_action, {})
        manual_params_str = st.text_area(
            "Parameters (JSON)",
            value=json.dumps(st.session_state.get("_manual_params", {}), indent=2),
            height=140, key="manual_params_text")
        if st.button("Apply", key="manual_apply_btn"):
            try:
                ep = json.loads(manual_params_str) if manual_params_str.strip() else {}
                if manual_action == "drop_column":
                    ep.setdefault("target_column", target_column)
                ep = resolve_action_columns(
                    manual_action, ep,
                    ep.get("columns") or ep.get("target_columns"),
                    st.session_state.df_current)
                df_after, diff, warnings = ACTION_EXECUTORS[manual_action](
                    st.session_state.df_current, ep)
                st.session_state.df_current = df_after
                st.session_state.provenance.add_step(log_step(
                    dataset_id=dataset_name,
                    step_id=len(getattr(st.session_state.provenance, "steps", [])),
                    action_name=manual_action, approved=True, status="applied",
                    params=ep, rationale="Manual action (no AI)",
                    before_schema={}, after_schema={},
                    diff_summary=diff, warnings=warnings, error=None,
                ))
                st.success("Applied")
                if warnings:
                    st.warning(str(warnings))
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ===============================================================================
# STEP 4 - EXPORT
# ===============================================================================
st.markdown('<div class="step-header"><span class="step-badge">4</span>'
            '<span class="step-title">Export &amp; Audit Trail</span></div>',
            unsafe_allow_html=True)

if st.session_state.step < 4 and not st.session_state.execution_done:
    st.info("Execute actions in Step 3 to unlock downloads.")
else:
    st.markdown(
        "Download everything you need: the cleaned dataset, the full audit log "
        "of every change made, and the AI plan for reproducibility.")

    safe_name = dataset_name.replace(" ", "_")
    prov_dict = st.session_state.provenance.to_dict()
    plan_dict = (
        st.session_state.last_plan.model_dump()
        if st.session_state.last_plan
        else {}
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            "Download original CSV",
            data=st.session_state.df_original.to_csv(index=False).encode(),
            file_name=f"{safe_name}_original.csv", mime="text/csv",
            use_container_width=True,
            help="Your dataset exactly as you uploaded it, untouched.")
    with c2:
        st.download_button(
            "Download cleaned CSV",
            data=st.session_state.df_current.to_csv(index=False).encode(),
            file_name=f"{safe_name}_cleaned.csv", mime="text/csv",
                use_container_width=True,
            help="The dataset after all approved transformations have been applied.")
    with c3:
        st.download_button(
            "Download audit log (JSON)",
            data=json.dumps(prov_dict, indent=2, default=str).encode(),
            file_name=f"{safe_name}_provenance.json", mime="application/json",
            use_container_width=True,
            help="Full provenance: every proposed action, decision, parameter and outcome.")
    with c4:
        st.download_button(
            "Download AI plan (JSON)",
            data=json.dumps(plan_dict, indent=2, default=str).encode(),
            file_name=f"{safe_name}_plan.json", mime="application/json",
            use_container_width=True,
            help="The structured cleaning plan proposed by the LLM, for reproducibility.")
