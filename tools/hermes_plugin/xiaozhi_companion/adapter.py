"""Hermes public Platform Adapter for a Xiaozhi-compatible companion."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gateway.config import Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

from .protocol import VoiceMessage, XiaozhiProtocolServer


PLATFORM_NAME = "xiaozhi_companion"


def _bearer_token(value: str) -> str:
    token = str(value or "").strip()
    if token and " " not in token:
        return f"Bearer {token}"
    return token


class XiaozhiCompanionAdapter(BasePlatformAdapter):
    """Bridge protocol sessions to the standard Hermes message boundary."""

    def __init__(self, config, *, session_id_factory=None):
        super().__init__(config=config, platform=Platform(PLATFORM_NAME))
        extra = getattr(config, "extra", {}) or {}
        token = _bearer_token(extra.get("token", ""))
        if not token:
            raise ValueError("xiaozhi_companion token is required")
        self._transport_mode = str(extra.get("transport", "")).strip().lower()
        self.protocol = XiaozhiProtocolServer(
            expected_token=token,
            on_voice_message=self._handle_voice_message,
            session_id_factory=session_id_factory,
        )

    @property
    def name(self) -> str:
        return "Xiaozhi Companion"

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        del is_reconnect
        if self._transport_mode != "loopback":
            return False
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        await self.protocol.disconnect_all()
        self._mark_disconnected()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        del reply_to, metadata
        session = self.protocol.active_session(str(chat_id))
        if session is None or not session.is_negotiated:
            return SendResult(success=False, error="companion session is not active")
        await session.send_transcript(content)
        return SendResult(success=True, message_id=session.session_id)

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
        source = self.build_source(
            chat_id=recording.device_id,
            chat_name="alain00",
            chat_type="dm",
            user_id=recording.device_id,
            user_name="alain00",
        )
        await self.handle_message(
            MessageEvent(
                text="[voice message]",
                message_type=MessageType.VOICE,
                source=source,
                user_id=recording.device_id,
                user_name="alain00",
                raw_message=recording,
                metadata={
                    "xiaozhi_session_id": recording.session_id,
                    "audio_format": recording.audio_params.format,
                    "sample_rate": recording.audio_params.sample_rate,
                    "channels": recording.audio_params.channels,
                    "frame_duration": recording.audio_params.frame_duration,
                },
            )
        )


def check_requirements() -> bool:
    return True


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(str(extra.get("token", "")).strip()) and (
        str(extra.get("transport", "")).strip().lower() == "loopback"
    )


def register(ctx) -> None:
    ctx.register_platform(
        name=PLATFORM_NAME,
        label="Xiaozhi Companion",
        adapter_factory=lambda config: XiaozhiCompanionAdapter(config),
        check_fn=check_requirements,
        validate_config=validate_config,
        required_env=[],
        install_hint="No additional Python packages are required for the protocol harness.",
        allowed_users_env="XIAOZHI_COMPANION_ALLOWED_DEVICES",
        allow_all_env="XIAOZHI_COMPANION_ALLOW_ALL_DEVICES",
        emoji="🫧",
        pii_safe=False,
        platform_hint=(
            "You are speaking through a small voice companion. "
            "Use concise, conversational responses."
        ),
    )
