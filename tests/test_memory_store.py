"""Per-agent memory stores — buffer cap, replay determinism, factory."""

from __future__ import annotations

import pytest

from scm_bench.memory.store import (
    BoundedBufferMemory,
    EpisodicMemory,
    StatelessMemory,
    make_memory,
)


def test_stateless_memory_never_remembers() -> None:
    mem = StatelessMemory()
    for i in range(10):
        mem.append(i)
    assert mem.snapshot() == []
    mem.clear()
    assert mem.snapshot() == []


def test_bounded_buffer_evicts_oldest_first() -> None:
    mem = BoundedBufferMemory(max_entries=3)
    for i in range(5):
        mem.append(i)
    assert mem.snapshot() == [2, 3, 4]


def test_bounded_buffer_clear_empties() -> None:
    mem = BoundedBufferMemory(max_entries=4)
    mem.append("a")
    mem.append("b")
    mem.clear()
    assert mem.snapshot() == []


def test_bounded_buffer_rejects_zero_or_negative_cap() -> None:
    with pytest.raises(ValueError):
        BoundedBufferMemory(max_entries=0)
    with pytest.raises(ValueError):
        BoundedBufferMemory(max_entries=-1)


def test_bounded_buffer_snapshot_is_a_copy() -> None:
    mem = BoundedBufferMemory(max_entries=4)
    mem.append(1)
    snap = mem.snapshot()
    snap.append(99)
    assert mem.snapshot() == [1]


def test_episodic_memory_truncates_to_cap() -> None:
    mem = EpisodicMemory(max_entries=2)
    mem.append(("tag", 1))
    mem.append(("tag", 2))
    mem.append(("tag", 3))
    assert mem.snapshot() == [("tag", 2), ("tag", 3)]


def test_episodic_memory_rejects_zero_or_negative_cap() -> None:
    with pytest.raises(ValueError):
        EpisodicMemory(max_entries=0)
    with pytest.raises(ValueError):
        EpisodicMemory(max_entries=-5)


def test_episodic_memory_snapshot_is_a_copy() -> None:
    mem = EpisodicMemory(max_entries=4)
    mem.append("x")
    snap = mem.snapshot()
    snap.clear()
    assert mem.snapshot() == ["x"]


def test_make_memory_factory_dispatches_by_mode() -> None:
    assert isinstance(make_memory("stateless"), StatelessMemory)
    assert isinstance(make_memory("bounded_buffer", max_entries=8), BoundedBufferMemory)
    assert isinstance(make_memory("episodic", max_entries=8), EpisodicMemory)


def test_make_memory_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        make_memory("not_a_mode")  # type: ignore[arg-type]


def test_replay_determinism_same_inputs_same_snapshot() -> None:
    """Two memories given the same append sequence return identical snapshots."""
    a = BoundedBufferMemory(max_entries=5)
    b = BoundedBufferMemory(max_entries=5)
    seq = [{"tick": i, "v": i * 2} for i in range(7)]
    for item in seq:
        a.append(item)
        b.append(item)
    assert a.snapshot() == b.snapshot()
