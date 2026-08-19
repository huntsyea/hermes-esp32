"""Transport-independent Xiaozhi WebSocket protocol v1 server core."""

from __future__ import annotations

import asyncio
import hmac
import inspect
import json
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Protocol


PROTOCOL_VERSION = 1
TRANSPORT = "websocket"


@dataclass(frozen=True)
class AudioParameters:
    format: str = "opus"
    sample_rate: int = 16000
    channels: int = 1
    frame_duration: int = 60

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frame_duration": self.frame_duration,
        }


REQUIRED_AUDIO_PARAMETERS = AudioParameters()


@dataclass(frozen=True)
class VoiceMessage:
    session_id: str
    device_id: str
    client_id: str
    audio_params: AudioParameters
    frames: tuple[bytes, ...]


@dataclass(frozen=True)
class ProtocolDiagnostic:
    code: str
    session_id: str | None = None


class SessionTransport(Protocol):
    async def send_json(self, message: Mapping[str, Any]) -> None: ...

    async def send_binary(self, payload: bytes) -> None: ...

    async def close(self, code: int, reason: str) -> None: ...


class ProtocolError(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class AuthenticationError(ProtocolError):
    pass


VoiceMessageHandler = Callable[[VoiceMessage], Any | Awaitable[Any]]
DiagnosticSink = Callable[[ProtocolDiagnostic], Any]
SessionIdFactory = Callable[[], str]
TokenResolver = Callable[[str], str | None]


def _header(headers: Mapping[str, str], name: str) -> str:
    target = name.casefold()
    for key, value in headers.items():
        if str(key).casefold() == target:
            return str(value).strip()
    return ""


class XiaozhiProtocolServer:
    """Own active sessions while remaining independent of WebSocket libraries."""

    def __init__(
        self,
        *,
        expected_token: str | None = None,
        token_resolver: TokenResolver | None = None,
        on_voice_message: VoiceMessageHandler,
        diagnostic_sink: DiagnosticSink | None = None,
        session_id_factory: SessionIdFactory | None = None,
        max_frame_bytes: int = 16 * 1024,
        max_recording_bytes: int = 4 * 1024 * 1024,
    ):
        if bool(expected_token) == bool(token_resolver):
            raise ValueError("exactly_one_token_source_required")
        self._token_resolver = token_resolver or (lambda _device_id: expected_token)
        self._on_voice_message = on_voice_message
        self._diagnostic_sink = diagnostic_sink
        self._session_id_factory = session_id_factory or (
            lambda: uuid.uuid4().hex
        )
        self._max_frame_bytes = max_frame_bytes
        self._max_recording_bytes = max_recording_bytes
        self._sessions_by_device: dict[str, XiaozhiSession] = {}
        self._open_lock = asyncio.Lock()

    async def open_session(
        self,
        transport: SessionTransport,
        headers: Mapping[str, str],
    ) -> "XiaozhiSession":
        if _header(headers, "Protocol-Version") != str(PROTOCOL_VERSION):
            self._diagnose("unsupported_version")
            raise ProtocolError("unsupported_version")

        device_id = _header(headers, "Device-Id")
        client_id = _header(headers, "Client-Id")
        if not device_id or not client_id:
            self._diagnose("identity_required")
            raise ProtocolError("identity_required")

        authorization = _header(headers, "Authorization")
        expected_token = self._token_resolver(device_id)
        comparison_token = expected_token or ("\0" * 32)
        token_matches = hmac.compare_digest(authorization, comparison_token)
        authenticated = bool(expected_token) and token_matches
        if not authenticated:
            self._diagnose("authentication_rejected")
            raise AuthenticationError("authentication_rejected")

        session = XiaozhiSession(
            server=self,
            transport=transport,
            session_id=self._session_id_factory(),
            device_id=device_id,
            client_id=client_id,
            max_frame_bytes=self._max_frame_bytes,
            max_recording_bytes=self._max_recording_bytes,
        )
        async with self._open_lock:
            previous = self._sessions_by_device.get(device_id)
            self._sessions_by_device[device_id] = session
        if previous is not None:
            await previous._invalidate(4001, "replaced")
        self._diagnose("session_opened", session.session_id)
        return session

    def active_session(self, device_id: str) -> "XiaozhiSession | None":
        session = self._sessions_by_device.get(device_id)
        if session is None or not session.is_active:
            return None
        return session

    async def disconnect_all(self) -> None:
        sessions = tuple(self._sessions_by_device.values())
        self._sessions_by_device.clear()
        for session in sessions:
            await session._invalidate(1001, "adapter_shutdown")

    async def _release(self, session: "XiaozhiSession") -> None:
        async with self._open_lock:
            if self._sessions_by_device.get(session.device_id) is session:
                self._sessions_by_device.pop(session.device_id, None)

    async def _emit_voice_message(self, message: VoiceMessage) -> None:
        result = self._on_voice_message(message)
        if inspect.isawaitable(result):
            await result

    def _diagnose(self, code: str, session_id: str | None = None) -> None:
        if self._diagnostic_sink is not None:
            self._diagnostic_sink(ProtocolDiagnostic(code, session_id))


class XiaozhiSession:
    def __init__(
        self,
        *,
        server: XiaozhiProtocolServer,
        transport: SessionTransport,
        session_id: str,
        device_id: str,
        client_id: str,
        max_frame_bytes: int,
        max_recording_bytes: int,
    ):
        self._server = server
        self._transport = transport
        self.session_id = session_id
        self.device_id = device_id
        self.client_id = client_id
        self._max_frame_bytes = max_frame_bytes
        self._max_recording_bytes = max_recording_bytes
        self._active = True
        self._negotiated = False
        self._listening = False
        self._tts_active = False
        self._frames: list[bytes] = []
        self._recording_bytes = 0

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_negotiated(self) -> bool:
        return self._negotiated

    async def receive_text(self, payload: str) -> None:
        self._assert_active()
        try:
            message = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            self._fail("malformed_json")
        if not isinstance(message, dict):
            self._fail("json_object_required")

        message_type = message.get("type")
        if not isinstance(message_type, str) or not message_type:
            self._fail("message_type_required")

        if not self._negotiated:
            if message_type != "hello":
                self._fail("hello_required")
            await self._negotiate(message)
            return

        self._assert_session_id(message)
        if message_type == "listen":
            await self._handle_listen(message)
            return
        if message_type == "hello":
            self._fail("duplicate_hello")
        self._fail("unsupported_message")

    async def receive_binary(self, payload: bytes) -> None:
        self._assert_ready()
        if not self._listening:
            self._fail("listen_not_active")
        if not isinstance(payload, bytes) or not payload:
            self._fail("invalid_audio_frame")
        if len(payload) > self._max_frame_bytes:
            self._fail("audio_frame_too_large")
        if self._recording_bytes + len(payload) > self._max_recording_bytes:
            self._fail("recording_too_large")
        self._frames.append(payload)
        self._recording_bytes += len(payload)

    async def send_transcript(self, text: str) -> None:
        await self._send_json("stt", text=str(text))

    async def send_emotion(self, emotion: str, *, text: str | None = None) -> None:
        fields: dict[str, Any] = {"emotion": str(emotion)}
        if text is not None:
            fields["text"] = str(text)
        await self._send_json("llm", **fields)

    async def start_tts(self) -> None:
        self._assert_ready()
        if self._tts_active:
            self._fail("tts_already_active")
        self._tts_active = True
        await self._send_json("tts", state="start")

    async def send_tts_sentence(self, text: str) -> None:
        if not self._tts_active:
            self._fail("tts_not_active")
        await self._send_json("tts", state="sentence_start", text=str(text))

    async def send_tts_audio(self, payload: bytes) -> None:
        self._assert_ready()
        if not self._tts_active:
            self._fail("tts_not_active")
        if not isinstance(payload, bytes) or not payload:
            self._fail("invalid_audio_frame")
        await self._transport.send_binary(payload)

    async def stop_tts(self) -> None:
        if not self._tts_active:
            self._fail("tts_not_active")
        await self._send_json("tts", state="stop")
        self._tts_active = False

    async def disconnect(self) -> None:
        if not self._active:
            return
        self._active = False
        self._discard_recording()
        self._tts_active = False
        await self._server._release(self)
        self._server._diagnose("session_disconnected", self.session_id)

    async def _negotiate(self, message: Mapping[str, Any]) -> None:
        if message.get("version") != PROTOCOL_VERSION:
            self._fail("unsupported_version")
        if message.get("transport") != TRANSPORT:
            self._fail("unsupported_transport")
        if message.get("audio_params") != REQUIRED_AUDIO_PARAMETERS.as_dict():
            self._fail("unsupported_audio_parameters")
        self._negotiated = True
        await self._transport.send_json(
            {
                "type": "hello",
                "version": PROTOCOL_VERSION,
                "transport": TRANSPORT,
                "session_id": self.session_id,
                "audio_params": REQUIRED_AUDIO_PARAMETERS.as_dict(),
            }
        )
        self._server._diagnose("session_negotiated", self.session_id)

    async def _handle_listen(self, message: Mapping[str, Any]) -> None:
        state = message.get("state")
        if state == "start":
            if self._listening:
                self._fail("listen_already_active")
            if message.get("mode") != "manual":
                self._fail("unsupported_listen_mode")
            self._listening = True
            self._frames.clear()
            self._recording_bytes = 0
            self._server._diagnose("listen_started", self.session_id)
            return
        if state == "stop":
            if not self._listening:
                self._fail("listen_not_active")
            frames = tuple(self._frames)
            self._listening = False
            self._frames.clear()
            self._recording_bytes = 0
            if not frames:
                self._fail("empty_recording")
            await self._server._emit_voice_message(
                VoiceMessage(
                    session_id=self.session_id,
                    device_id=self.device_id,
                    client_id=self.client_id,
                    audio_params=REQUIRED_AUDIO_PARAMETERS,
                    frames=frames,
                )
            )
            self._server._diagnose("voice_message_emitted", self.session_id)
            return
        self._fail("unsupported_listen_state")

    async def _send_json(self, message_type: str, **fields: Any) -> None:
        self._assert_ready()
        await self._transport.send_json(
            {"session_id": self.session_id, "type": message_type, **fields}
        )

    async def _invalidate(self, code: int, reason: str) -> None:
        if not self._active:
            return
        self._active = False
        self._discard_recording()
        self._tts_active = False
        await self._transport.close(code, reason)
        self._server._diagnose("session_invalidated", self.session_id)

    def _assert_active(self) -> None:
        if not self._active:
            self._fail("stale_session")

    def _assert_ready(self) -> None:
        self._assert_active()
        if not self._negotiated:
            self._fail("hello_required")

    def _assert_session_id(self, message: Mapping[str, Any]) -> None:
        if message.get("session_id") != self.session_id:
            self._fail("session_mismatch")

    def _discard_recording(self) -> None:
        self._listening = False
        self._frames.clear()
        self._recording_bytes = 0

    def _fail(self, code: str) -> None:
        self._server._diagnose(code, self.session_id)
        raise ProtocolError(code)
