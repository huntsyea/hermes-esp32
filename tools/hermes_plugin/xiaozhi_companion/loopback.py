"""Deterministic, explicit loopback transport for the protocol harness."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CloseEvent:
    code: int
    reason: str


class LoopbackTransport:
    def __init__(self):
        self.json_messages: list[dict[str, Any]] = []
        self.binary_messages: list[bytes] = []
        self.close_events: list[CloseEvent] = []

    async def send_json(self, message: Mapping[str, Any]) -> None:
        self.json_messages.append(dict(message))

    async def send_binary(self, payload: bytes) -> None:
        self.binary_messages.append(payload)

    async def close(self, code: int, reason: str) -> None:
        self.close_events.append(CloseEvent(code, reason))

