"""OpenEuroLLM evaluation tasks packaged for the lm-eval harness.

The actual task definitions live in the bundled ``tasks`` data directory.
Use :func:`tasks_path` to obtain its location at runtime regardless of how
the package was installed (wheel, editable, or ``uv tool``).
"""

from importlib.resources import files
from importlib.resources.abc import Traversable

__all__ = ["tasks_path"]


def tasks_path() -> Traversable:
    """Return the path to the bundled ``tasks`` directory."""
    return files("oellm_tasks") / "tasks"
