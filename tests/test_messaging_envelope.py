"""Message envelope round-trip, role validation, type enum."""

from __future__ import annotations

import json

import pytest

from scm_bench.messaging.envelope import Message
from scm_bench.messaging.types import (
    ALL_MESSAGE_TYPES,
    VALID_ROLES,
    MessageType,
)


def test_message_roundtrip_via_json() -> None:
    msg = Message(
        receiver_role="wholesaler",
        type=MessageType.FORECAST,
        payload={"horizon": 4, "values": [3, 4, 5, 4]},
    )
    raw = msg.model_dump_json()
    revived = Message.model_validate_json(raw)
    assert revived.receiver_role == "wholesaler"
    assert revived.type == MessageType.FORECAST
    assert revived.payload == {"horizon": 4, "values": [3, 4, 5, 4]}
    assert revived.message_id == msg.message_id


def test_message_id_is_unique_per_construction() -> None:
    a = Message(receiver_role="factory", type=MessageType.STATUS)
    b = Message(receiver_role="factory", type=MessageType.STATUS)
    assert a.message_id != b.message_id


def test_message_rejects_unknown_role() -> None:
    with pytest.raises(ValueError):
        Message(receiver_role="transporter", type=MessageType.STATUS)


def test_message_rejects_extra_fields() -> None:
    with pytest.raises(ValueError):
        Message(
            receiver_role="retailer",
            type=MessageType.STATUS,
            unexpected="boom",
        )


def test_message_payload_default_is_empty_dict() -> None:
    msg = Message(receiver_role="retailer", type=MessageType.ALERT)
    assert msg.payload == {}


def test_all_message_types_are_in_enum() -> None:
    expected = {"forecast", "status", "intent", "alert", "request", "response"}
    assert {m.value for m in MessageType} == expected
    assert ALL_MESSAGE_TYPES == frozenset(MessageType)


def test_valid_roles_match_four_echelon_chain() -> None:
    assert VALID_ROLES == frozenset(
        {"retailer", "wholesaler", "distributor", "factory"}
    )


def test_message_serialised_payload_is_pure_json() -> None:
    msg = Message(
        receiver_role="distributor",
        type=MessageType.INTENT,
        payload={"order": 7, "reason": "smoothing"},
    )
    data = json.loads(msg.model_dump_json())
    assert data["payload"]["order"] == 7
    assert data["type"] == "intent"


def test_message_type_enum_round_trip_via_string() -> None:
    msg = Message(receiver_role="factory", type="request")  # str coerces to enum
    assert msg.type == MessageType.REQUEST


def test_message_rejects_non_enum_type() -> None:
    with pytest.raises(ValueError):
        Message(receiver_role="factory", type="not_a_type")
