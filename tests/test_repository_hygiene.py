from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_repository_hygiene import hygiene_errors, tracked_paths  # noqa: E402


def test_tracked_tree_passes_repository_hygiene_policy() -> None:
    paths = tracked_paths()
    assert len(paths) < 500
    assert hygiene_errors(paths) == []
