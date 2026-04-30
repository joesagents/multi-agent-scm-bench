"""Message types for the A2A-style supply chain bench protocol."""

from enum import Enum


class MessageType(str, Enum):
    FORECAST = "forecast"
    STATUS = "status"
    INTENT = "intent"
    ALERT = "alert"
    REQUEST = "request"
    RESPONSE = "response"


ALL_MESSAGE_TYPES: frozenset[MessageType] = frozenset(MessageType)


VALID_ROLES: frozenset[str] = frozenset(
    {"retailer", "wholesaler", "distributor", "factory"}
)
