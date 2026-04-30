"""`scm_bench export-template` overwrite-safety tests.

`--force` previously did an unconditional `shutil.rmtree(out)` on whatever path the user passed.
A path typo could wipe arbitrary directories. The fix is a marker file
+ a manifest-shape probe; this test pins both behaviours.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from scm_bench import SDK_VERSION
from scm_bench.cli import (
    BUNDLE_MARKER_FILENAME,
    LEGACY_BUNDLE_MARKER_FILENAME,
    app,
)

runner = CliRunner()


def test_export_template_drops_marker(tmp_path: Path) -> None:
    out = tmp_path / "team_bundle"
    result = runner.invoke(app, ["export-template", "--out", str(out)])
    assert result.exit_code == 0, result.output
    marker = out / BUNDLE_MARKER_FILENAME
    assert marker.exists()
    payload = json.loads(marker.read_text())
    assert payload["created_by"] == "scm_bench export-template"
    assert payload["sdk_version"] == SDK_VERSION


def test_force_overwrite_refuses_unrelated_dir(tmp_path: Path) -> None:
    """A path that is NOT a scm_bench bundle must not be rmtree'd."""
    out = tmp_path / "important_user_data"
    out.mkdir()
    (out / "thesis.md").write_text("# do not delete\n")

    result = runner.invoke(app, ["export-template", "--out", str(out), "--force"])
    assert result.exit_code == 1
    assert "refusing --force overwrite" in result.output
    # User data must survive.
    assert (out / "thesis.md").exists()


def test_force_overwrite_allowed_with_marker(tmp_path: Path) -> None:
    out = tmp_path / "team_bundle"
    runner.invoke(app, ["export-template", "--out", str(out)])
    assert (out / BUNDLE_MARKER_FILENAME).exists()
    # Now simulate a re-export with --force on top of the existing bundle.
    result = runner.invoke(app, ["export-template", "--out", str(out), "--force"])
    assert result.exit_code == 0, result.output
    assert (out / BUNDLE_MARKER_FILENAME).exists()


def test_force_overwrite_allowed_with_legacy_manifest(tmp_path: Path) -> None:
    """A bundle created before markers existed (manifest.json present, no marker)
    is still recognised and overwrite-safe."""
    out = tmp_path / "legacy_bundle"
    out.mkdir()
    (out / "manifest.json").write_text(
        json.dumps({"team_id": "legacy", "sdk_version": SDK_VERSION})
    )
    result = runner.invoke(app, ["export-template", "--out", str(out), "--force"])
    assert result.exit_code == 0, result.output


def test_force_overwrite_allowed_with_legacy_beergame_marker(tmp_path: Path) -> None:
    """A bundle dir created by the previous CLI name (`.beergame-bundle`
    marker) is still recognised by `--force` for one transition release."""
    out = tmp_path / "old_bundle"
    out.mkdir()
    (out / LEGACY_BUNDLE_MARKER_FILENAME).write_text(
        json.dumps({"created_by": "beergame export-template"})
    )
    result = runner.invoke(app, ["export-template", "--out", str(out), "--force"])
    assert result.exit_code == 0, result.output
    # New bundle now carries the new marker (and the old one is gone after rmtree).
    assert (out / BUNDLE_MARKER_FILENAME).exists()
    assert not (out / LEGACY_BUNDLE_MARKER_FILENAME).exists()


def test_no_force_refuses_when_dir_exists(tmp_path: Path) -> None:
    out = tmp_path / "team_bundle"
    out.mkdir()
    result = runner.invoke(app, ["export-template", "--out", str(out)])
    assert result.exit_code == 1
    assert "use --force" in result.output
