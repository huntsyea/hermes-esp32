"""Desktop protocol harness for one simulated Xiaozhi device."""

from __future__ import annotations

import json
from typing import Mapping

from .loopback import LoopbackTransport
from .protocol import XiaozhiProtocolServer, XiaozhiSession


class ProtocolHarness:
    def __init__(
        self,
        server: XiaozhiProtocolServer,
        headers: Mapping[str, str],
    ):
        self._server = server
        self._headers = dict(headers)
        self.transport = LoopbackTransport()
        self.session: XiaozhiSession | None = None

    async def connect(self) -> XiaozhiSession:
        self.session = await self._server.open_session(
            self.transport, self._headers
        )
        return self.session

    async def negotiate(self) -> None:
        session = self._require_session()
        await session.receive_text(
            json.dumps(
                {
                    "type": "hello",
                    "version": 1,
                    "features": {"mcp": True},
                    "transport": "websocket",
                    "audio_params": {
                        "format": "opus",
                        "sample_rate": 16000,
                        "channels": 1,
                        "frame_duration": 60,
                    },
                }
            )
        )

    async def start_listening(self) -> None:
        session = self._require_session()
        await session.receive_text(
            json.dumps(
                {
                    "session_id": session.session_id,
                    "type": "listen",
                    "state": "start",
                    "mode": "manual",
                }
            )
        )

    async def send_audio(self, *frames: bytes) -> None:
        session = self._require_session()
        for frame in frames:
            await session.receive_binary(frame)

    async def stop_listening(self) -> None:
        session = self._require_session()
        await session.receive_text(
            json.dumps(
                {
                    "session_id": session.session_id,
                    "type": "listen",
                    "state": "stop",
                }
            )
        )

    async def disconnect(self) -> None:
        if self.session is not None:
            await self.session.disconnect()

    def _require_session(self) -> XiaozhiSession:
        if self.session is None:
            raise RuntimeError("harness_not_connected")
        return self.session

