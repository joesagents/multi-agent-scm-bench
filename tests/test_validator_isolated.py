"""Subprocess-isolated validator tests.

The in-process validator is fine for trusted code, but `test-bundle`
runs untrusted agent bundles. These tests build deliberately hostile
bundle skeletons and confirm the parent process survives, with the
right stable error code.
"""

from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

import pytest

from scm_bench import SDK_VERSION
from scm_bench.sdk.validator import validate_bundle_isolated


def _starter_bundle(tmp_path: Path) -> Path:
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


def _replace_role_agent(bundle: Path, role: str, body: str) -> None:
    """Overwrite one role's agent.py with `body` (must define class Agent)
    and rewrite agent.yaml entrypoint to point at `Agent`."""
    (bundle / role / "agent.py").write_text(textwrap.dedent(body))
    yaml_path = bundle / role / "agent.yaml"
    text = yaml_path.read_text()
    new_lines = []
    for line in text.splitlines():
        if line.startswith("entrypoint:"):
            new_lines.append("entrypoint: agent.py:Agent")
        else:
            new_lines.append(line)
    yaml_path.write_text("\n".join(new_lines) + "\n")


def test_isolated_happy_path(tmp_path: Path) -> None:
    bundle = _starter_bundle(tmp_path)
    payload = validate_bundle_isolated(bundle, smoke_ticks=3, timeout_s=30)
    assert payload["ok"] is True, payload
    assert payload["smoke_ticks"] == 3


def test_isolated_catches_infinite_loop(tmp_path: Path) -> None:
    """A bundle that hangs in step() must surface as E_BUNDLE_TIMEOUT,
    not freeze the parent forever."""
    bundle = _starter_bundle(tmp_path)
    _replace_role_agent(
        bundle,
        "retailer",
        '''
        from scm_bench.sdk.agent import Agent as _Base
        from scm_bench.sdk.contract import AgentDecision

        class Agent(_Base):
            def step(self, observation, inbox, t):
                while True:
                    pass
        ''',
    )
    payload = validate_bundle_isolated(bundle, smoke_ticks=2, timeout_s=3)
    assert payload["ok"] is False
    assert payload["code"] == "E_BUNDLE_TIMEOUT", payload


def test_isolated_catches_sysexit(tmp_path: Path) -> None:
    """A bundle that calls sys.exit() in __init__ must NOT take down the
    parent — surfaces as E_BUNDLE_CRASH or a normal validation failure
    (the worker catches BaseException and reports E_BUNDLE_CRASH from
    inside; either is acceptable)."""
    bundle = _starter_bundle(tmp_path)
    _replace_role_agent(
        bundle,
        "retailer",
        '''
        import sys
        from scm_bench.sdk.agent import Agent as _Base
        from scm_bench.sdk.contract import AgentDecision

        class Agent(_Base):
            def __init__(self):
                sys.exit(7)

            def step(self, observation, inbox, t):
                return AgentDecision(order_qty=0)
        ''',
    )
    payload = validate_bundle_isolated(bundle, smoke_ticks=2, timeout_s=10)
    assert payload["ok"] is False, payload
    assert payload["code"] in {"E_BUNDLE_CRASH", "E_AGENT_INSTANTIATION"}


def test_isolated_catches_baseexception(tmp_path: Path) -> None:
    """Raising BaseException (which Exception-only handlers miss) is
    caught at the subprocess boundary."""
    bundle = _starter_bundle(tmp_path)
    _replace_role_agent(
        bundle,
        "retailer",
        '''
        from scm_bench.sdk.agent import Agent as _Base
        from scm_bench.sdk.contract import AgentDecision

        class Agent(_Base):
            def step(self, observation, inbox, t):
                raise BaseException("boom")
        ''',
    )
    payload = validate_bundle_isolated(bundle, smoke_ticks=2, timeout_s=10)
    assert payload["ok"] is False, payload
    assert payload["code"] in {"E_BUNDLE_CRASH", "E_SMOKE_RUN_FAILED"}


def test_isolated_propagates_normal_validation_errors(tmp_path: Path) -> None:
    """A bundle with a missing manifest still produces the in-process
    error code, not E_BUNDLE_CRASH."""
    nonexistent = tmp_path / "no_such_dir"
    payload = validate_bundle_isolated(nonexistent, smoke_ticks=1, timeout_s=10)
    assert payload["ok"] is False
    assert payload["code"] == "E_BUNDLE_ROOT_MISSING"
