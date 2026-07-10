"""Build and independently verify the pre-cleanup scientific artifact archive."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs/manifests/repository_inventory.csv"
LOCAL_VERIFICATION = ROOT / "docs/manifests/archive_verification.json"
LOCAL_FILE_LIST = ROOT / "docs/manifests/archive_file_list.txt"
STEM = "IMBH_legacy_steady_dae_artifacts_pre-cleanup-p0-2026-07-11"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_selected() -> list[dict[str, str]]:
    with INVENTORY.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row["proposed_action"] in {"ARCHIVE", "DELETE"}]


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _validate_member_names(archive: tarfile.TarFile) -> None:
    for member in archive.getmembers():
        candidate = Path(member.name)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise RuntimeError(f"unsafe archive member: {member.name}")
        if not member.isfile():
            raise RuntimeError(f"unexpected non-file archive member: {member.name}")


def main() -> None:
    destination = Path(
        os.environ.get(
            "IMBH_LEGACY_ARCHIVE_DIR",
            str(ROOT.parent / "IMBH_QPE_legacy_archive_2026-07-11"),
        )
    ).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows = _load_selected()
    expected = {row["path"]: row for row in rows}
    archive_path = destination / f"{STEM}.tar.gz"
    manifest_path = destination / f"{STEM}_MANIFEST.csv"
    sums_path = destination / f"{STEM}_SHA256SUMS.txt"
    verification_path = destination / f"{STEM}_VERIFICATION.json"

    started = time.time()
    for relative, row in expected.items():
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(relative)
        if source.stat().st_size != int(row["bytes"]):
            raise RuntimeError(f"size changed since inventory: {relative}")
        if _sha256(source) != row["sha256"]:
            raise RuntimeError(f"hash changed since inventory: {relative}")

    _write_manifest(manifest_path, rows)
    LOCAL_FILE_LIST.write_text("".join(f"{path}\n" for path in sorted(expected)))

    with tarfile.open(archive_path, mode="w:gz", compresslevel=6) as archive:
        for relative in sorted(expected):
            archive.add(ROOT / relative, arcname=relative, recursive=False)

    archive_sha = _sha256(archive_path)
    manifest_sha = _sha256(manifest_path)
    sums_path.write_text(
        f"{archive_sha}  {archive_path.name}\n"
        f"{manifest_sha}  {manifest_path.name}\n"
    )

    with tempfile.TemporaryDirectory(prefix="imbh_archive_verify_", dir=destination) as temp:
        extract_root = Path(temp)
        with tarfile.open(archive_path, mode="r:gz") as archive:
            _validate_member_names(archive)
            members = archive.getmembers()
            archive.extractall(extract_root, members=members)
        extracted = sorted(
            path.relative_to(extract_root).as_posix()
            for path in extract_root.rglob("*")
            if path.is_file()
        )
        if extracted != sorted(expected):
            missing = sorted(set(expected) - set(extracted))
            extra = sorted(set(extracted) - set(expected))
            raise RuntimeError(f"archive membership mismatch; missing={missing}, extra={extra}")
        for relative, row in expected.items():
            extracted_path = extract_root / relative
            if extracted_path.stat().st_size != int(row["bytes"]):
                raise RuntimeError(f"extracted size mismatch: {relative}")
            if _sha256(extracted_path) != row["sha256"]:
                raise RuntimeError(f"extracted hash mismatch: {relative}")

    verification = {
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "selected_file_count": len(rows),
        "selected_uncompressed_bytes": sum(int(row["bytes"]) for row in rows),
        "source_commit": "0a000767a915880c0710b8f4ec03eb0c64aa168a",
        "source_tag": "pre-cleanup-p0-2026-07-11",
        "verification": "PASS: extracted every member and matched size and SHA-256",
        "elapsed_seconds": time.time() - started,
    }
    text = json.dumps(verification, indent=2, sort_keys=True) + "\n"
    verification_path.write_text(text)
    LOCAL_VERIFICATION.write_text(text)
    shutil.copy2(INVENTORY, destination / "repository_inventory_full.csv")
    print(text, end="")


if __name__ == "__main__":
    main()
