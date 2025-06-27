"""
Centralised project-relative path management for Mind2Web2.

Typical usage
-------------
from pathlib import Path
from mind2web2.utils.path_config import PathConfig

project_root = Path(__file__).resolve().parents[2]   # adapt as needed
paths = PathConfig(project_root)

# Override anything you like:
paths.apply_overrides(cache_root=Path("/tmp/my_cache"))

print(paths.eval_scripts_root)
print(paths.default_script_for("task_001"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PathConfig:
    """
    Holds every project-relative directory in one place.

    All attributes are absolute `Path` objects and never contain `~`.
    """
    project_root: Path

    # Dataset subtree
    dataset_root: Path = field(init=False)
    answers_root: Path = field(init=False)
    answers_sample_one_root: Path = field(init=False)
    eval_scripts_root: Path = field(init=False)
    tasks_root: Path = field(init=False)
    eval_results_root: Path = field(init=False)
    cache_root: Path = field(init=False)

    # Workspace subtree
    workspace_root: Path = field(init=False)
    ws_cache_root: Path = field(init=False)
    ws_eval_results_root: Path = field(init=False)
    ws_eval_scripts_root: Path = field(init=False)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        self.project_root = self.project_root.expanduser().resolve()

        # Dataset
        self.dataset_root = self.project_root / "dataset"
        self.answers_root = self.dataset_root / "answers"
        self.answers_sample_one_root = self.dataset_root / "answers_sample_one"

        self.eval_scripts_root = self.dataset_root / "eval_scripts"
        self.tasks_root = self.dataset_root / "tasks"
        self.eval_results_root = self.dataset_root / "eval_results"

        # self.cache_root = self.dataset_root / "cache"

        # Workspace
        self.workspace_root = self.project_root / "workspace"
        self.ws_cache_root = self.workspace_root / "cache"
        self.ws_eval_results_root = self.workspace_root / "eval_results"
        self.ws_eval_scripts_root = self.workspace_root / "eval_scripts"

        # Scripts
        self.scripts_root = self.project_root / "scripts"
        self.run_eval_script = self.scripts_root / "run_eval.py"



    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def default_script_for(self, task_id: str) -> Path:
        """Return `<eval_scripts_root>/<task_id>.py`."""
        return self.eval_scripts_root / f"{task_id}.py"

    def apply_overrides(
        self,
        *,
        answers_root: Optional[Path] = None,
        ws_cache_root: Optional[Path] = None,
        output_root: Optional[Path] = None,
        eval_scripts_root: Optional[Path] = None,
    ) -> None:
        """
        Overwrite selected directories in-place.
        All arguments are absolute or will be resolved/expanded.
        """
        if answers_root is not None:
            self.answers_root = answers_root.expanduser().resolve()
        if ws_cache_root is not None:
            self.ws_cache_root = ws_cache_root.expanduser().resolve()
        if output_root is not None:
            self.ws_eval_results_root = output_root.expanduser().resolve()
        if eval_scripts_root is not None:
            self.eval_scripts_root = eval_scripts_root.expanduser().resolve()

    # ------------------------------------------------------------------ #
    # Debug helpers
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # pragma: no cover
        fields = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({fields})"
