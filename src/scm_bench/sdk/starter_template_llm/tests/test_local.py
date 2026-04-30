"""Local smoke test — equivalent to `scm_bench test-bundle .`.

Run from the bundle root:

    pytest tests/test_local.py -v

This loads the bundle through the validator and runs a 5-tick smoke
simulation on `intro_step_demand`. If this passes, your bundle is
structurally valid and will be accepted by the operator's evaluator.

This test does NOT require Ollama to be running. The LLM-driven step()
falls back to the mirror policy when the backend is unreachable, so
the smoke test always completes. To verify your policy actually calls
the model, set `SCB_LLM_BACKEND` and watch tokens_used > 0 on a real
run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scm_bench.sdk.validator import ValidationError, validate_bundle

BUNDLE_ROOT = Path(__file__).resolve().parent.parent


def test_bundle_validates() -> None:
    try:
        report = validate_bundle(BUNDLE_ROOT, smoke_ticks=5)
    except ValidationError as e:
        pytest.fail(f"bundle validation failed [{e.code}]: {e}")
    assert report.smoke_run_ok, f"smoke run did not complete: {report.smoke_error}"
