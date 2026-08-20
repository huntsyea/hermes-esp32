"""Hermes public Platform Adapter for a Xiaozhi-compatible companion."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
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

from .network import (
    FileTokenResolver,
    ListenerConfig,
    ListenerPolicyError,
    WebsocketListener,
    build_server_ssl_context,
)
from .ogg_opus import packets_to_ogg_opus
from .protocol import ProtocolError, VoiceMessage, XiaozhiProtocolServer, XiaozhiSession
from .response_text import response_preview
from .tts_audio import load_tts_opus_packets


PLATFORM_NAME = "xiaozhi_companion"


@dataclass
class VoiceResponseContext:
    session: XiaozhiSession
    speech_text: str = ""
    tts_text_sent: bool = False
    interaction_generation: int | None = None

    def __post_init__(self) -> None:
        if self.interaction_generation is None:
            self.interaction_generation = self.session.interaction_generation


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
        try:
            if is_final_response:
                if not self._context_is_current(context, session):
                    return SendResult(success=False, error="companion response session is stale")
                if not context.tts_text_sent:
                    await session.send_emotion(
                        "neutral",
                        text=response_preview(content),
                        generation=context.interaction_generation,
                    )
                return SendResult(success=True, message_id=session.session_id)
            generation = None
            if context is not None:
                if not self._context_is_current(context, session):
                    return SendResult(success=False, error="companion response session is stale")
                generation = context.interaction_generation
            await session.send_transcript(content, generation=generation)
            return SendResult(success=True, message_id=session.session_id)
        except ProtocolError:
            return SendResult(success=False, error="companion response session is stale")

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
        if not self._context_is_current(context, session):
            return SendResult(success=False, error="companion response session is stale")
        if not context.speech_text:
            return SendResult(success=False, error="companion response text is empty")
        try:
            packets = await asyncio.to_thread(
                self._tts_audio_loader,
                audio_path,
                ffmpeg_executable=self._ffmpeg_executable,
            )
            if not self._context_is_current(context, session):
                return SendResult(success=False, error="companion response session is stale")
            await session.start_tts(generation=context.interaction_generation)
            await session.send_tts_sentence(
                response_preview(context.speech_text),
                generation=context.interaction_generation,
            )
            context.tts_text_sent = True
            for packet in packets:
                await session.send_tts_audio(
                    packet, generation=context.interaction_generation
                )
            await session.stop_tts(generation=context.interaction_generation)
            return SendResult(success=True, message_id=session.session_id)
        except Exception as exc:
            if self._context_is_current(context, session):
                try:
                    await session.send_error(
                        "tts", generation=context.interaction_generation
                    )
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
        session = self.protocol.active_session(recording.device_id)
        if (
            session is None
            or session.session_id != recording.session_id
            or session.interaction_generation != recording.interaction_generation
        ):
            return
        try:
            ogg_audio = await asyncio.to_thread(
                packets_to_ogg_opus,
                recording.frames,
                serial=self._ogg_serial_factory(),
                input_sample_rate=recording.audio_params.sample_rate,
                channels=recording.audio_params.channels,
                frame_duration_ms=recording.audio_params.frame_duration,
            )
            if not self._recording_is_current(recording, session):
                return
            cache_task = asyncio.create_task(
                asyncio.to_thread(self._audio_cache, ogg_audio, ext=".ogg")
            )
            try:
                cached_audio = await asyncio.shield(cache_task)
            except asyncio.CancelledError:
                cache_task.add_done_callback(self._discard_cache_task_result)
                raise
            if not self._recording_is_current(recording, session):
                self._discard_cached_audio(cached_audio)
                return
        except Exception:
            if session.interaction_generation == recording.interaction_generation:
                try:
                    await session.send_error(
                        "stt", generation=recording.interaction_generation
                    )
                except Exception:
                    pass
            return
        source = self.build_source(
            chat_id=recording.device_id,
            chat_name="alain00",
            chat_type="dm",
            user_id=recording.device_id,
            user_name="alain00",
        )
        context = VoiceResponseContext(
            session, interaction_generation=recording.interaction_generation
        )
        context_token = self._voice_response_context.set(context)
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
        except Exception:
            if self._context_is_current(context, session):
                try:
                    await session.send_error(
                        "agent", generation=context.interaction_generation
                    )
                except Exception:
                    pass
        finally:
            self._voice_response_context.reset(context_token)

    @staticmethod
    def _discard_cached_audio(path: str) -> None:
        try:
            Path(path).unlink(missing_ok=True)
        except (OSError, TypeError, ValueError):
            pass

    def _discard_cache_task_result(self, task: asyncio.Task[str]) -> None:
        try:
            cached_audio = task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            return
        self._discard_cached_audio(cached_audio)

    def _recording_is_current(
        self, recording: VoiceMessage, session: XiaozhiSession
    ) -> bool:
        return bool(
            self.protocol.active_session(recording.device_id) is session
            and session.session_id == recording.session_id
            and session.interaction_generation == recording.interaction_generation
        )

    def _context_is_current(
        self,
        context: VoiceResponseContext | None,
        session: XiaozhiSession | None,
    ) -> bool:
        return bool(
            context is not None
            and session is not None
            and context.session is session
            and self.protocol.active_session(session.device_id) is session
            and context.interaction_generation == session.interaction_generation
        )


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
            listener_config = ListenerConfig.from_extra(extra)
            build_server_ssl_context(listener_config)
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
