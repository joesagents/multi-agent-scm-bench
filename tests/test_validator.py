"""Bundle validator — happy path + malformed bundles.

Each malformed case must surface the documented stable error code so the
grader can triage at a glance.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scm_bench import SDK_VERSION
from scm_bench.sdk.validator import (
    ValidationError,
    validate_bundle,
)


@pytest.fixture
def fresh_bundle(tmp_path: Path) -> Path:
    """Copy the starter template into a fresh tmp dir and return its root."""
    from importlib import resources

    template_root = resources.files("scm_bench.sdk").joinpath(
        "starter_template"
    )
    dest = tmp_path / "team_bundle"
    with resources.as_file(template_root) as template_path:
        shutil.copytree(template_path, dest)

    manifest_path = dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sdk_version"] = SDK_VERSION
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for role in ("retailer", "wholesaler", "distributor", "factory"):
        agent_yaml = dest / role / "agent.yaml"
        text = agent_yaml.read_text()
        text = text.replace('sdk_version: "1.0.0"', f'sdk_version: "{SDK_VERSION}"')
        agent_yaml.write_text(text)
    return dest


def test_happy_path(fresh_bundle: Path) -> None:
    report = validate_bundle(fresh_bundle, smoke_ticks=5)
    assert report.smoke_run_ok
    assert set(report.agents) == {"retailer", "wholesaler", "distributor", "factory"}
    assert report.smoke_run_record is not None
    assert len(report.smoke_run_record.ticks) == 5


def test_root_missing(tmp_path: Path) -> None:
    nonexistent = tmp_path / "no_such_dir"
    with pytest.raises(ValidationError) as excinfo:
        validate_bundle(nonexistent)
    assert excinfo.value.code == "E_BUNDLE_ROOT_MISSING"


def test_team_manifest_missing(fresh_bundle: Path) -> None:
    (fresh_bundle / "manifest.json").unlink()
    with pytest.raises(ValidationError) as excinfo:
        validate_bundle(fresh_bundle)
    assert excinfo.value.code == "E_TEAM_MANIFEST_MISSING"


def test_team_manifest_invalid_json(fresh_bundle: Path) -> None:
    (fresh_bundle / "manifest.json").write_text("{not valid json")
    with pytest.raises(ValidationError) as excinfo:
        validate_bundle(fresh_bundle)
    assert excinfo.value.code == "E_TEAM_MANIFEST_INVALID_JSON"


def test_team_manifest_missing_role(fresh_bundle: Path) -> None:
    manifest = json.loads((fresh_bundle / "manifest.json").read_text())
    del manifest["agents"]["factory"]
    (fresh_bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValidationError) as excinfo:
        validate_bundle(fresh_bundle)
    assert excinfo.value.code == "E_TEAM_MANIFEST_SCHEMA"


def test_team_manifest_extra_role(fresh_bundle: Path) -> None:
    manifest = json.loads((fresh_bundle / "manifest.json").read_text())
    manifest["agents"]["transporter"] = "transporter"
    (fresh_bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValidationError) as excinfo:
        validate_bundle(fresh_bundle)
    assert excinfo.value.code == "E_TEAM_MANIFEST_SCHEMA"


def test_sdk_version_mismatch_team(fresh_bundle: Path) -> None:
    manifest = json.loads((fresh_bundle / "manifest.json").read_text())
    manifest["sdk_version"] = "999.0.0"
    (fresh_bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValidationError) as excinfo:
        validate_bundle(fresh_bundle)
    assert excinfo.value.code == "E_SDK_VERSION_MISMATCH"


def test_agent_dir_missing(fresh_bundle: Path) -> None:
    shutil.rmtree(fresh_bundle / "wholesaler")
    with pytest.raises(ValidationError) as excinfo:
        validate_bundle(fresh_bundle)
    assert excinfo.value.code == "E_AGENT_DIR_MISSING"


def test_agent_manifest_missing(fresh_bundle: Path) -> None:
    (fresh_bundle / "distributor" / "agent.yaml").unlink()
    with pytest.raises(ValidationError) as excinfo:
        validate_bundle(fresh_bundle)
    assert excinfo.value.code == "E_AGENT_MANIFEST_MISSING"


def test_agent_role_mismatch(fresh_bundle: Path) -> None:
    yaml_path = fresh_bundle / "retailer" / "agent.yaml"
    text = yaml_path.read_text().replace("role: retailer", "role: factory")
    yaml_path.write_text(text)
    with pytest.raises(ValidationError) as excinfo:
        validate_bundle(fresh_bundle)
    assert excinfo.value.code == "E_AGENT_ROLE_MISMATCH"


def test_agent_entrypoint_module_missing(fresh_bundle: Path) -> None:
    (fresh_bundle / "factory" / "agent.py").unlink()
    with pytest.raises(ValidationError) as excinfo:
        validate_bundle(fresh_bundle)
    assert excinfo.value.code == "E_AGENT_ENTRYPOINT_MISSING"


def test_agent_entrypoint_class_missing(fresh_bundle: Path) -> None:
    yaml_path = fresh_bundle / "wholesaler" / "agent.yaml"
    text = yaml_path.read_text().replace(
        "entrypoint: agent.py:WholesalerAgent",
        "entrypoint: agent.py:NoSuchClass",
    )
    yaml_path.write_text(text)
    with pytest.raises(ValidationError) as excinfo:
        validate_bundle(fresh_bundle)
    assert excinfo.value.code == "E_AGENT_ENTRYPOINT_CLASS_MISSING"


def test_agent_entrypoint_not_agent(fresh_bundle: Path) -> None:
    py_path = fresh_bundle / "distributor" / "agent.py"
    py_path.write_text("class DistributorAgent:\n    pass\n")
    with pytest.raises(ValidationError) as excinfo:
        validate_bundle(fresh_bundle)
    assert excinfo.value.code == "E_AGENT_ENTRYPOINT_NOT_AGENT"
