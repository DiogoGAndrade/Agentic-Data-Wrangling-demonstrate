# engine/schemas.py

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ActionType(str, Enum):
    cast_type = "cast_type"
    handle_missing = "handle_missing"
    normalize_text = "normalize_text"
    encode_categorical = "encode_categorical"
    encode_categorical_per_column = "encode_categorical_per_column"  # C4
    drop_column = "drop_column"
    remove_outliers = "remove_outliers"
    clip_outliers = "clip_outliers"      # C4 alias
    select_features = "select_features"  # C4
    bin_numeric = "bin_numeric"          # C4
    deduplicate = "deduplicate"
    fix_column_names = "fix_column_names"

class Severity(str, Enum):
    info = "info"
    warning = "warning"
    critical = "critical"

class Diagnostic(BaseModel):
    code: str
    severity: Severity
    message: str
    evidence: Optional[Dict[str, Any]] = Field(default_factory=dict)

class Action(BaseModel):
    action: ActionType
    rationale: str = Field(...)
    target_columns: List[str] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)

class Prognostic(BaseModel):
    code: str
    severity: Severity
    message: str
    evidence: Dict[str, Any] = Field(default_factory=dict)

class LLMPlan(BaseModel):
    dataset_summary: str = ""
    reasoning_steps: List[str] = Field(default_factory=list)
    actions: List[Action] = Field(default_factory=list)
    diagnostics: List[Diagnostic] = Field(default_factory=list)
