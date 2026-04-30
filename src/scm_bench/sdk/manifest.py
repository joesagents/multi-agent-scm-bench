"""Bundle and per-agent manifests.

Two manifest files per bundle:
- team_bundle/manifest.json       — TeamManifest (JSON, machine-stamped by export-template)
- team_bundle/<role>/agent.yaml   — AgentManifest (YAML, agent-edited)

The validator loads both, fails closed on any schema violation, and
attaches a stable error code to every failure for diagnostic triage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from scm_bench.messaging.types import MessageType, VALID_ROLES

ROLES_ORDER: tuple[str, ...] = ("retailer", "wholesaler", "distributor", "factory")


class ManifestError(Exception):
    """Raised when a manifest file cannot be loaded or fails schema validation."""

    def __init__(self, code: str, message: str, *, path: Path | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.path = path


MemoryMode = Literal["stateless", "bounded_buffer", "episodic"]


class AgentManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    agent_name: str
    entrypoint: str  # "agent.py:Agent"
    memory_mode: MemoryMode = "stateless"
    memory_max_entries: int = Field(default=32, ge=1, le=4096)
    supports_tools: list[str] = Field(default_factory=list)
    supports_messages: list[MessageType] = Field(default_factory=list)
    sdk_version: str

    @field_validator("role")
    @classmethod
    def _role_known(cls, v: str) -> str:
        if v not in VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(VALID_ROLES)}, got {v!r}")
        return v

    @field_validator("entrypoint")
    @classmethod
    def _entrypoint_format(cls, v: str) -> str:
        if ":" not in v:
            raise ValueError("entrypoint must be of the form 'module.py:ClassName'")
        module_part, _, class_part = v.partition(":")
        if not module_part.endswith(".py") or not class_part.isidentifier():
            raise ValueError(
                "entrypoint must be 'module.py:ClassName' "
                "(module ends in .py, class is a Python identifier)"
            )
        return v

    @property
    def entrypoint_module(self) -> str:
        return self.entrypoint.split(":", 1)[0]

    @property
    def entrypoint_class(self) -> str:
        return self.entrypoint.split(":", 1)[1]


class TeamManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    team_id: str
    team_name: str
    submission_version: str
    sdk_version: str
    agents: dict[str, str]  # role -> directory name (usually role itself)

    @field_validator("agents")
    @classmethod
    def _agents_complete(cls, v: dict[str, str]) -> dict[str, str]:
        keys = set(v.keys())
        expected = set(ROLES_ORDER)
        missing = expected - keys
        extra = keys - expected
        if missing:
            raise ValueError(
                f"team manifest missing agents for roles: {sorted(missing)}"
            )
        if extra:
            raise ValueError(
                f"team manifest has unexpected agent roles: {sorted(extra)}"
            )
        return v


def load_team_manifest(bundle_root: Path) -> TeamManifest:
    path = bundle_root / "manifest.json"
    if not path.exists():
        raise ManifestError(
            "E_TEAM_MANIFEST_MISSING",
            f"team manifest not found at {path}",
            path=path,
        )
    try:
        raw: Any = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ManifestError(
            "E_TEAM_MANIFEST_INVALID_JSON",
            f"team manifest is not valid JSON: {e}",
            path=path,
        ) from e
    try:
        return TeamManifest.model_validate(raw)
    except Exception as e:
        raise ManifestError(
            "E_TEAM_MANIFEST_SCHEMA",
            f"team manifest schema error: {e}",
            path=path,
        ) from e


def load_agent_manifest(bundle_root: Path, role: str, role_dir_name: str) -> AgentManifest:
    path = bundle_root / role_dir_name / "agent.yaml"
    if not path.exists():
        raise ManifestError(
            "E_AGENT_MANIFEST_MISSING",
            f"agent manifest not found at {path}",
            path=path,
        )
    try:
        raw: Any = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ManifestError(
            "E_AGENT_MANIFEST_INVALID_YAML",
            f"agent manifest is not valid YAML: {e}",
            path=path,
        ) from e
    if not isinstance(raw, dict):
        raise ManifestError(
            "E_AGENT_MANIFEST_NOT_OBJECT",
            f"agent manifest at {path} must be a YAML mapping",
            path=path,
        )
    try:
        manifest = AgentManifest.model_validate(raw)
    except Exception as e:
        raise ManifestError(
            "E_AGENT_MANIFEST_SCHEMA",
            f"agent manifest schema error at {path}: {e}",
            path=path,
        ) from e
    if manifest.role != role:
        raise ManifestError(
            "E_AGENT_ROLE_MISMATCH",
            f"agent at {path} declares role={manifest.role!r} but lives in {role!r} dir",
            path=path,
        )
    return manifest
