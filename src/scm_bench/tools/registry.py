"""Local tool registry.

Phase 1 ships three builtin tools and a registry; the validator checks
declared `supports_tools` against the registry. Phase 2 enforces that an
agent only invokes tools it declared.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ToolFn = Callable[..., Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        if name in self._tools:
            raise ValueError(f"tool {name!r} already registered")
        self._tools[name] = fn

    def get(self, name: str) -> ToolFn:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name!r}")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
