"""Hermes public Platform Adapter for a Xiaozhi-compatible companion."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
import secrets
import shutil
from typing import Any, Callable, Dict, Optional

from gateway.config import Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_audio_from_bytes,
)

from .network import FileTokenResolver, ListenerConfig, ListenerPolicyError, WebsocketListener
from .ogg_opus import packets_to_ogg_opus
from .protocol import VoiceMessage, XiaozhiProtocolServer, XiaozhiSession
from .response_text import response_preview
from .tts_audio import load_tts_opus_packets


PLATFORM_NAME = "xiaozhi_companion"


@dataclass
class VoiceResponseContext:
    session: XiaozhiSession
    speech_text: str = ""
    tts_text_sent: bool = False


def _bearer_token(value: str) -> str:
    token = str(value or "").strip()
    if token and " " not in token:
        return f"Bearer {token}"
    return token


class XiaozhiCompanionAdapter(BasePlatformAdapter):
    """Bridge protocol sessions to the standard Hermes message boundary."""

    def __init__(
        self,
        config,
        *,
        session_id_factory=None,
        ogg_serial_factory: Callable[[], int] | None = None,
        audio_cache: Callable[..., str] | None = None,
        tts_audio_loader: Callable[..., tuple[bytes, ...]] | None = None,
    ):
        super().__init__(config=config, platform=Platform(PLATFORM_NAME))
        self._ogg_serial_factory = ogg_serial_factory or (
            lambda: secrets.randbits(32)
        )
        self._audio_cache = audio_cache or cache_audio_from_bytes
        self._tts_audio_loader = tts_audio_loader or load_tts_opus_packets
        self._voice_response_context: ContextVar[VoiceResponseContext | None] = ContextVar(
            "xiaozhi_voice_response", default=None
        )
        extra = getattr(config, "extra", {}) or {}
        self._ffmpeg_executable = str(
            extra.get("ffmpeg_executable", shutil.which("ffmpeg") or "ffmpeg")
        )
        self._transport_mode = str(extra.get("transport", "")).strip().lower()
        self._listener = None
        if self._transport_mode == "wss":
            listener_config = ListenerConfig.from_extra(extra)
            token_resolver = FileTokenResolver(extra.get("device_tokens_file", ""))
            self.protocol = XiaozhiProtocolServer(
                token_resolver=token_resolver,
                on_voice_message=self._handle_voice_message,
                session_id_factory=session_id_factory,
            )
            self._listener = WebsocketListener(listener_config, self.protocol)
        elif self._transport_mode == "loopback":
            token = _bearer_token(extra.get("token", ""))
            if not token:
                raise ValueError("xiaozhi_companion token is required")
            self.protocol = XiaozhiProtocolServer(
                expected_token=token,
                on_voice_message=self._handle_voice_message,
                session_id_factory=session_id_factory,
            )
            if bool(extra.get("listener_enabled", False)):
                self._listener = WebsocketListener(
                    ListenerConfig.from_extra(extra), self.protocol
                )
        else:
            raise ListenerPolicyError("transport_must_be_wss_or_loopback")

    @property
    def name(self) -> str:
        return "Xiaozhi Companion"

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        del is_reconnect
        if self._listener is not None:
            await self._listener.start()
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        if self._listener is not None:
            await self._listener.stop()
        await self.protocol.disconnect_all()
        self._mark_disconnected()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        del reply_to
        session = self.protocol.active_session(str(chat_id))
        if session is None or not session.is_negotiated:
            return SendResult(success=False, error="companion session is not active")
        context = self._voice_response_context.get()
        is_final_response = bool(metadata and metadata.get("notify"))
        if is_final_response:
            if context is None or context.session is not session:
                return SendResult(success=False, error="companion response session is stale")
            if not context.tts_text_sent:
                await session.send_emotion("neutral", text=response_preview(content))
            return SendResult(success=True, message_id=session.session_id)
        await session.send_transcript(content)
        return SendResult(success=True, message_id=session.session_id)

    def prepare_tts_text(self, text: str) -> str:
        speech_text = super().prepare_tts_text(text)
        context = self._voice_response_context.get()
        if context is not None:
            context.speech_text = speech_text
        return speech_text

    async def play_tts(self, chat_id: str, audio_path: str, **kwargs) -> SendResult:
        del kwargs
        context = self._voice_response_context.get()
        session = self.protocol.active_session(str(chat_id))
        if context is None or session is None or context.session is not session:
            return SendResult(success=False, error="companion response session is stale")
        if not context.speech_text:
            return SendResult(success=False, error="companion response text is empty")
        lifecycle_started = False
        try:
            packets = await asyncio.to_thread(
                self._tts_audio_loader,
                audio_path,
                ffmpeg_executable=self._ffmpeg_executable,
            )
            if self.protocol.active_session(str(chat_id)) is not session:
                return SendResult(success=False, error="companion response session is stale")
            await session.start_tts()
            lifecycle_started = True
            await session.send_tts_sentence(response_preview(context.speech_text))
            context.tts_text_sent = True
            for packet in packets:
                await session.send_tts_audio(packet)
            await session.stop_tts()
            return SendResult(success=True, message_id=session.session_id)
        except Exception as exc:
            if lifecycle_started and self.protocol.active_session(str(chat_id)) is session:
                try:
                    await session.stop_tts()
                except Exception:
                    pass
            return SendResult(success=False, error=f"companion tts failed: {type(exc).__name__}")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        del chat_id, metadata

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        del chat_id, image_url, caption, reply_to, metadata
        return SendResult(success=False, error="images are not supported")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {
            "name": f"Companion {chat_id}",
            "type": "dm",
            "chat_id": str(chat_id),
        }

    async def _handle_voice_message(self, recording: VoiceMessage) -> None:
        ogg_audio = packets_to_ogg_opus(
            recording.frames,
            serial=self._ogg_serial_factory(),
            input_sample_rate=recording.audio_params.sample_rate,
            channels=recording.audio_params.channels,
            frame_duration_ms=recording.audio_params.frame_duration,
        )
        cached_audio = self._audio_cache(ogg_audio, ext=".ogg")
        source = self.build_source(
            chat_id=recording.device_id,
            chat_name="alain00",
            chat_type="dm",
            user_id=recording.device_id,
            user_name="alain00",
        )
        session = self.protocol.active_session(recording.device_id)
        if session is None or session.session_id != recording.session_id:
            return
        context_token = self._voice_response_context.set(VoiceResponseContext(session))
        try:
            await self.handle_message(MessageEvent(
                text="[voice message]",
                message_type=MessageType.VOICE,
                source=source,
                user_id=recording.device_id,
                user_name="alain00",
                media_urls=[cached_audio],
                media_types=["audio/ogg"],
                raw_message=recording,
                metadata={
                    "xiaozhi_session_id": recording.session_id,
                    "audio_format": recording.audio_params.format,
                    "sample_rate": recording.audio_params.sample_rate,
                    "channels": recording.audio_params.channels,
                    "frame_duration": recording.audio_params.frame_duration,
                },
            ))
        finally:
            self._voice_response_context.reset(context_token)


def check_requirements() -> bool:
    try:
        import websockets

        major = int(websockets.__version__.split(".", 1)[0])
        return major >= 15
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    try:
        transport = str(extra.get("transport", "")).strip().lower()
        if transport == "loopback":
            if not str(extra.get("token", "")).strip():
                return False
            if bool(extra.get("listener_enabled", False)):
                ListenerConfig.from_extra(extra)
            return True
        if transport == "wss":
            ListenerConfig.from_extra(extra)
            FileTokenResolver(extra.get("device_tokens_file", ""))
            return True
        return False
    except (ListenerPolicyError, OSError, ValueError):
        return False


def register(ctx) -> None:
    ctx.register_platform(
        name=PLATFORM_NAME,
        label="Xiaozhi Companion",
        adapter_factory=lambda config: XiaozhiCompanionAdapter(config),
        check_fn=check_requirements,
        validate_config=validate_config,
        required_env=[],
        install_hint="Requires websockets 15 or newer for the production WSS listener.",
        allowed_users_env="XIAOZHI_COMPANION_ALLOWED_DEVICES",
        allow_all_env="XIAOZHI_COMPANION_ALLOW_ALL_DEVICES",
        emoji="🫧",
        pii_safe=False,
        platform_hint=(
            "You are speaking through a small voice companion. "
            "Use concise, conversational responses."
        ),
    )
