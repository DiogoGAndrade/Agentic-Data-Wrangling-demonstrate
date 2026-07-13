# engine/provenance.py
# Code comments in English (per your preference)

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


class ProvenanceLog:
    """
    Canonical provenance interface used by:
      - Streamlit app (app/main.py)
      - evaluation scripts (evaluation/prepare_conditions.py, evaluation/run_experiments.py)

    Required:
      - self.steps: List[dict]
      - add_step(step: dict) -> None
      - to_dict() -> dict
    """

    def __init__(self) -> None:
        self.steps: List[Dict[str, Any]] = []

    def add_step(self, step: Dict[str, Any]) -> None:
        self.steps.append(step)

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": self.steps}


def log_step(
    *,
    dataset_id: str,
    step_id: int,
    action_name: str,
    approved: bool,
    status: str,  # applied | skipped | failed
    params: Dict[str, Any],
    rationale: str,
    before_schema: Dict[str, str],
    after_schema: Optional[Dict[str, str]],
    diff_summary: Optional[Dict[str, Any]],
    warnings: Optional[list],
    error: Optional[str],
) -> Dict[str, Any]:
    """
    IMPORTANT: field names must match what the UI / scripts read.
    In the Streamlit "Execution summary" you used:
      - step_id, action_name, status, warnings, diff_summary
    So we store those exact keys.
    """
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "dataset_id": dataset_id,
        "step_id": step_id,
        "action_name": action_name,
        "approved": approved,
        "status": status,
        "params": params,
        "rationale": rationale,
        "before_schema": before_schema,
        "after_schema": after_schema,
        "diff_summary": diff_summary,
        "warnings": warnings,
        "error": error,
    }
