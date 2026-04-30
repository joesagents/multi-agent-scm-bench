"""Per-agent local memory.

Phase 1 ships three memory modes — stateless, bounded_buffer, episodic —
but enforces only the buffer cap. Phase 2 adds quota auditing and a
violation surface.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Literal, Protocol

MemoryMode = Literal["stateless", "bounded_buffer", "episodic"]


class MemoryStore(Protocol):
    mode: MemoryMode

    def append(self, item: Any) -> None: ...
    def snapshot(self) -> list[Any]: ...
    def clear(self) -> None: ...


class StatelessMemory:
    mode: MemoryMode = "stateless"

    def append(self, item: Any) -> None:
        return None

    def snapshot(self) -> list[Any]:
        return []

    def clear(self) -> None:
        return None


class BoundedBufferMemory:
    mode: MemoryMode = "bounded_buffer"

    def __init__(self, max_entries: int = 32) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._buf: deque[Any] = deque(maxlen=max_entries)

    def append(self, item: Any) -> None:
        self._buf.append(item)

    def snapshot(self) -> list[Any]:
        return list(self._buf)

    def clear(self) -> None:
        self._buf.clear()


class EpisodicMemory:
    """Bounded list of structured (tag, payload) records."""

    mode: MemoryMode = "episodic"

    def __init__(self, max_entries: int = 64) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._max = max_entries
        self._entries: list[Any] = []

    def append(self, item: Any) -> None:
        self._entries.append(item)
        if len(self._entries) > self._max:
            self._entries = self._entries[-self._max :]

    def snapshot(self) -> list[Any]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()


def make_memory(mode: MemoryMode, max_entries: int = 32) -> MemoryStore:
    if mode == "stateless":
        return StatelessMemory()
    if mode == "bounded_buffer":
        return BoundedBufferMemory(max_entries=max_entries)
    if mode == "episodic":
        return EpisodicMemory(max_entries=max_entries)
    raise ValueError(f"unknown memory mode: {mode!r}")
