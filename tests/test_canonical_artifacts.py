from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical"
MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_every_canonical_case_has_provenance_and_valid_checksums() -> None:
    cases = sorted(path for path in CANONICAL.iterdir() if path.is_dir())
    assert len(cases) == 8
    for case in cases:
        provenance = json.loads((case / "provenance.json").read_text())
        assert provenance["source_tag"] == "pre-cleanup-p0-2026-07-11"
        assert provenance["scientific_status"] in {
            "CERTIFIED",
            "SUPPORTED BUT NOT FULLY CERTIFIED",
            "DIAGNOSTIC ONLY",
            "REJECTED",
        }
        for line in (case / "SHA256SUMS.txt").read_text().splitlines():
            expected, filename = line.split("  ", 1)
            assert _sha256(case / filename) == expected


def test_canonical_manifest_matches_files() -> None:
    with MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 52
    for row in rows:
        path = ROOT / row["path"]
        assert path.is_file()
        assert path.stat().st_size == int(row["bytes"])
        assert _sha256(path) == row["sha256"]


def test_canonical_numerical_anchors_are_accepted() -> None:
    paths = [
        CANONICAL / "no_wind_mdot5/state.npz",
        CANONICAL / "stream_no_wind_mdot2_fs080/state.npz",
        CANONICAL / "phase_dae_entry_N164/state.npz",
    ]
    for path in paths[:2]:
        with np.load(path, allow_pickle=False) as data:
            assert bool(data["accepted"])

    with np.load(paths[2], allow_pickle=False) as data:
        assert not bool(data["accepted"])
        assert int(data["n_nodes"]) == 164

    with np.load(paths[0], allow_pickle=False) as data:
        assert float(data["ratio"]) == 5.0
        assert int(data["n_nodes"]) == 768
    with np.load(paths[1], allow_pickle=False) as data:
        assert float(data["ratio"]) == 2.0
        assert float(data["stream_source_fraction"]) == 0.8
        assert int(data["n_nodes"]) == 896
