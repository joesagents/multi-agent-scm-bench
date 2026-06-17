"""Subprocess entrypoint for sandboxed bundle validation.

Invoked by `validator.validate_bundle_isolated` as a fresh Python
process so that a misbehaving agent bundle (infinite loop, OOM,
`sys.exit`, segfault, raised `BaseException`) cannot take down the
parent evaluator. The parent enforces wall-clock and (on POSIX)
RLIMIT_CPU / RLIMIT_AS limits via `preexec_fn`.

Wire format (single line of JSON on stdout):
- on success: {"ok": true, ...validate_bundle_safe payload}
- on validation error: {"ok": false, "code": "...", "message": "...", "path": "..."}
- on internal worker crash: nonzero exit code + stderr; the parent
  surfaces this as E_BUNDLE_CRASH.

Not directly user-facing — `scm-bench test-bundle` (and any other
caller that passes untrusted bundles) should call
`validate_bundle_isolated`, not this script.
"""

from __future__ import annotations

import json
import sys

from scm_bench.sdk.validator import validate_bundle_safe


def _usage() -> int:
    sys.stderr.write(
        "usage: python -m scm_bench.sdk.validator_worker "
        "<bundle_path> [smoke_ticks]\n"
    )
    return 2


def main(argv: list[str]) -> int:
    if not argv or len(argv) > 2:
        return _usage()
    bundle_path = argv[0]
    smoke_ticks = int(argv[1]) if len(argv) == 2 else 5
    payload = validate_bundle_safe(bundle_path, smoke_ticks=smoke_ticks)
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
