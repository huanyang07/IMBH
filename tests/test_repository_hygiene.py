from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_repository_hygiene import (  # noqa: E402
    MAX_TRACKED_FILES,
    hygiene_errors,
    tracked_paths,
)


def test_tracked_tree_passes_repository_hygiene_policy() -> None:
    paths = tracked_paths()
    assert len(paths) < MAX_TRACKED_FILES
    assert hygiene_errors(paths) == []
