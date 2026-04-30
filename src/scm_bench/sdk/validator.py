"""Bundle validator.

Loaded by the `scm_bench test-bundle` CLI command and by the agent-side
`tests/test_local.py`. Validates a bundle directory against the v2
contract and runs a short smoke simulation.

Stages (each fails closed with a stable error code):
1. Bundle root exists and is a directory.
2. team manifest loads + schema-validates (E_TEAM_MANIFEST_*).
3. For each role, the role directory exists.
4. Per-agent manifest loads + schema-validates (E_AGENT_MANIFEST_*).
5. SDK-version pin matches between team and agent manifests.
6. Per-agent entrypoint imports and resolves to a class subclassing Agent.
7. Class instantiates with no args.
8. Smoke run: instantiate all four, run `intro_step_demand` for
   `smoke_ticks` ticks, confirm each agent returned an AgentDecision,
   and enforce the manifest contract every tick (supports_messages,
   memory_max_entries — see runner/contract.py).

Returns a `ValidationReport` on success; raises `ValidationError` on
the first hard failure. Stable error codes are documented in
the source for diagnostic triage.

Threat model
------------
Two entry points, two trust levels:

- `validate_bundle()` imports and runs the bundle in the *current*
  Python process. Use this only on code you trust (built-in tests,
  the starter template). A malicious or buggy bundle can hang,
  exhaust memory, or call `sys.exit()` / raise `BaseException` and
  take the evaluator down with it.

- `validate_bundle_isolated()` spawns a fresh Python subprocess to
  run the import + smoke run, with a wall-clock timeout and (on
  POSIX) RLIMIT_CPU / RLIMIT_AS caps. A timeout becomes
  `E_BUNDLE_TIMEOUT`; any subprocess crash (segfault, signal,
  unhandled `BaseException`, OOM-killed) becomes `E_BUNDLE_CRASH`.
  This is the entry point the `scm_bench test-bundle` CLI uses by
  default. It is process isolation, not a sandbox: the child still
  has full filesystem and network access. For evaluation at scale,
  layer this under stronger isolation (cgroups, containers, etc.).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scm_bench import SDK_VERSION
from scm_bench.runner.contract import ContractViolationError
from scm_bench.runner.harness import RunRecord, run_one
from scm_bench.scenarios import builtin as scenarios_builtin
from scm_bench.sdk.agent import Agent
from scm_bench.sdk.contract import AgentDecision
from scm_bench.sdk.manifest import (
    ROLES_ORDER,
    AgentManifest,
    ManifestError,
    TeamManifest,
    load_agent_manifest,
    load_team_manifest,
)


class ValidationError(Exception):
    """Raised on the first hard validation failure."""

    def __init__(self, code: str, message: str, *, path: Path | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.path = path

    def __str__(self) -> str:
        base = super().__str__()
        if self.path is not None:
            return f"{base} (at {self.path})"
        return base


@dataclass
class ValidationReport:
    bundle_root: Path
    team: TeamManifest
    agents: dict[str, AgentManifest] = field(default_factory=dict)
    smoke_run_ok: bool = False
    smoke_error: str | None = None
    smoke_run_record: RunRecord | None = None


def _import_entrypoint(
    bundle_root: Path,
    role: str,
    role_dir: str,
    manifest: AgentManifest,
) -> type[Agent]:
    module_file = bundle_root / role_dir / manifest.entrypoint_module
    if not module_file.exists():
        raise ValidationError(
            "E_AGENT_ENTRYPOINT_MISSING",
            f"entrypoint module {manifest.entrypoint_module!r} not found",
            path=module_file,
        )

    mod_name = f"_bundle_{uuid.uuid4().hex}_{role}"
    spec = importlib.util.spec_from_file_location(mod_name, module_file)
    if spec is None or spec.loader is None:
        raise ValidationError(
            "E_AGENT_ENTRYPOINT_IMPORT",
            f"could not build import spec for {module_file}",
            path=module_file,
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ValidationError(
            "E_AGENT_ENTRYPOINT_IMPORT",
            f"importing {module_file} raised {type(e).__name__}: {e}",
            path=module_file,
        ) from e

    cls = getattr(module, manifest.entrypoint_class, None)
    if cls is None:
        raise ValidationError(
            "E_AGENT_ENTRYPOINT_CLASS_MISSING",
            f"class {manifest.entrypoint_class!r} not found in {module_file}",
            path=module_file,
        )
    if not isinstance(cls, type) or not issubclass(cls, Agent):
        raise ValidationError(
            "E_AGENT_ENTRYPOINT_NOT_AGENT",
            f"class {manifest.entrypoint_class!r} in {module_file} "
            "must subclass scm_bench.sdk.Agent",
            path=module_file,
        )
    return cls


def _instantiate(cls: type[Agent], role: str) -> Agent:
    try:
        return cls()
    except Exception as e:
        raise ValidationError(
            "E_AGENT_INSTANTIATION",
            f"instantiating {cls.__name__} for role={role!r} raised "
            f"{type(e).__name__}: {e}",
        ) from e


def validate_bundle(
    bundle_root: Path | str,
    *,
    smoke_ticks: int = 5,
    smoke_seed: int = 0,
) -> ValidationReport:
    root = Path(bundle_root).resolve()
    if not root.exists() or not root.is_dir():
        raise ValidationError(
            "E_BUNDLE_ROOT_MISSING",
            f"bundle root {root} does not exist or is not a directory",
            path=root,
        )

    try:
        team = load_team_manifest(root)
    except ManifestError as e:
        raise ValidationError(e.code, str(e), path=e.path) from e

    if team.sdk_version != SDK_VERSION:
        raise ValidationError(
            "E_SDK_VERSION_MISMATCH",
            f"team manifest sdk_version={team.sdk_version!r} does not match "
            f"installed SDK_VERSION={SDK_VERSION!r}",
            path=root / "manifest.json",
        )

    report = ValidationReport(bundle_root=root, team=team)

    agent_classes: dict[str, type[Agent]] = {}
    for role in ROLES_ORDER:
        role_dir = team.agents[role]
        role_path = root / role_dir
        if not role_path.is_dir():
            raise ValidationError(
                "E_AGENT_DIR_MISSING",
                f"role directory {role_dir!r} (for role={role!r}) "
                f"does not exist under {root}",
                path=role_path,
            )
        try:
            am = load_agent_manifest(root, role, role_dir)
        except ManifestError as e:
            raise ValidationError(e.code, str(e), path=e.path) from e

        if am.sdk_version != team.sdk_version:
            raise ValidationError(
                "E_SDK_VERSION_MISMATCH",
                f"agent {role!r} sdk_version={am.sdk_version!r} does not match "
                f"team sdk_version={team.sdk_version!r}",
                path=root / role_dir / "agent.yaml",
            )

        report.agents[role] = am
        agent_classes[role] = _import_entrypoint(root, role, role_dir, am)

    instances: dict[str, Agent] = {
        role: _instantiate(agent_classes[role], role) for role in ROLES_ORDER
    }

    scenario = scenarios_builtin.INTRO_STEP_DEMAND
    try:
        run_record = run_one(
            agents=instances,
            scenario=scenario,
            seed=smoke_seed,
            run_id=f"smoke-{uuid.uuid4().hex[:8]}",
            max_ticks=smoke_ticks,
            agent_manifests=dict(report.agents),
        )
    except ContractViolationError as e:
        report.smoke_error = f"{type(e).__name__}: {e}"
        raise ValidationError(
            e.code,
            f"contract violation during smoke run: {e}",
        ) from e
    except Exception as e:
        report.smoke_error = f"{type(e).__name__}: {e}"
        raise ValidationError(
            "E_SMOKE_RUN_FAILED",
            f"smoke run on {scenario.id!r} raised {type(e).__name__}: {e}",
        ) from e

    if not run_record.ticks:
        raise ValidationError(
            "E_SMOKE_RUN_FAILED",
            "smoke run produced no ticks",
        )

    for tick in run_record.ticks:
        for role, decision in tick.decisions.items():
            if not isinstance(decision, AgentDecision):
                raise ValidationError(
                    "E_DECISION_TYPE",
                    f"agent {role!r} returned {type(decision).__name__} "
                    "at tick {tick.t}, expected AgentDecision",
                )

    report.smoke_run_ok = True
    report.smoke_run_record = run_record
    return report


def _format_error_dict(e: ValidationError) -> dict[str, Any]:
    return {
        "ok": False,
        "code": e.code,
        "message": str(e),
        "path": str(e.path) if e.path else None,
    }


def validate_bundle_safe(
    bundle_root: Path | str,
    *,
    smoke_ticks: int = 5,
) -> dict[str, Any]:
    """Non-raising wrapper that returns a JSON-serialisable report dict.

    NOT process-isolated — runs in the caller's Python process. For
    untrusted bundles use `validate_bundle_isolated`.
    """
    try:
        report = validate_bundle(bundle_root, smoke_ticks=smoke_ticks)
    except ValidationError as e:
        return _format_error_dict(e)
    except BaseException as e:  # noqa: BLE001 — last-line defence
        # Catch BaseException so SystemExit / KeyboardInterrupt raised
        # by a hostile bundle don't escape the validator. Re-raise only
        # if the parent isolation wrapper is meant to see it.
        return {
            "ok": False,
            "code": "E_BUNDLE_CRASH",
            "message": f"bundle raised {type(e).__name__}: {e}",
            "path": None,
        }
    return {
        "ok": True,
        "team_id": report.team.team_id,
        "team_name": report.team.team_name,
        "sdk_version": report.team.sdk_version,
        "agents": {
            role: {
                "agent_name": am.agent_name,
                "memory_mode": am.memory_mode,
                "supports_tools": am.supports_tools,
                "supports_messages": [m.value for m in am.supports_messages],
            }
            for role, am in report.agents.items()
        },
        "smoke_ticks": len(report.smoke_run_record.ticks)
        if report.smoke_run_record
        else 0,
    }


DEFAULT_SANDBOX_TIMEOUT_S: float = 60.0
DEFAULT_SANDBOX_MEM_LIMIT_MB: int = 1024
DEFAULT_SANDBOX_CPU_LIMIT_S: int = 60


def _posix_rlimits(*, mem_mb: int, cpu_s: int):
    """Build a `preexec_fn` that caps memory + CPU on POSIX. Best-effort.

    Returns None on platforms that don't support `resource.setrlimit`
    (e.g. Windows) — the wall-clock timeout still applies.
    """
    try:
        import resource
    except ImportError:  # Windows
        return None

    def _apply() -> None:
        try:
            mem_bytes = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        except (ValueError, OSError):
            pass

    return _apply


def validate_bundle_isolated(
    bundle_root: Path | str,
    *,
    smoke_ticks: int = 5,
    timeout_s: float = DEFAULT_SANDBOX_TIMEOUT_S,
    mem_limit_mb: int = DEFAULT_SANDBOX_MEM_LIMIT_MB,
    cpu_limit_s: int = DEFAULT_SANDBOX_CPU_LIMIT_S,
) -> dict[str, Any]:
    """Spawn a worker subprocess to validate `bundle_root` under quotas.

    Returns the same dict shape as `validate_bundle_safe`. Adds two
    stable error codes the in-process variant cannot produce:

    - E_BUNDLE_TIMEOUT — wall-clock budget exceeded.
    - E_BUNDLE_CRASH   — child process died (signal, segfault, OOM,
      unhandled BaseException, malformed worker output).

    On POSIX, RLIMIT_AS / RLIMIT_CPU caps are applied via preexec_fn.
    """
    cmd = [
        sys.executable,
        "-m",
        "scm_bench.sdk.validator_worker",
        str(Path(bundle_root)),
        str(smoke_ticks),
    ]
    preexec = _posix_rlimits(mem_mb=mem_limit_mb, cpu_s=cpu_limit_s) if os.name == "posix" else None
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            preexec_fn=preexec,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "code": "E_BUNDLE_TIMEOUT",
            "message": f"validation worker exceeded {timeout_s}s wall clock",
            "path": str(bundle_root),
        }
    except OSError as e:
        return {
            "ok": False,
            "code": "E_BUNDLE_CRASH",
            "message": f"could not spawn validator worker: {e}",
            "path": str(bundle_root),
        }

    stdout = proc.stdout.strip()
    if not stdout:
        signal_part = ""
        if proc.returncode is not None and proc.returncode < 0:
            signal_part = f" signal={-proc.returncode}"
        tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
        return {
            "ok": False,
            "code": "E_BUNDLE_CRASH",
            "message": f"validator worker died (exit={proc.returncode}{signal_part}): {tail}",
            "path": str(bundle_root),
        }
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as e:
        return {
            "ok": False,
            "code": "E_BUNDLE_CRASH",
            "message": f"validator worker emitted non-JSON output: {e}",
            "path": str(bundle_root),
        }
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "code": "E_BUNDLE_CRASH",
            "message": "validator worker output was not a JSON object",
            "path": str(bundle_root),
        }
    return payload
