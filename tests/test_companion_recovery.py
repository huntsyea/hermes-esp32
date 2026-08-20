import asyncio
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_PARENT = ROOT
HERMES_ROOT = Path(
    os.environ.get("HERMES_AGENT_ROOT", Path.home() / ".hermes" / "hermes-agent")
)
HERMES_PYTHON = HERMES_ROOT / "venv" / "bin" / "python"
sys.path.insert(0, str(PLUGIN_PARENT))


class XiaozhiCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from xiaozhi_companion.protocol import XiaozhiProtocolServer

        self.voice_messages = []
        self.server = XiaozhiProtocolServer(
            expected_token="Bearer test-token",
            on_voice_message=self.voice_messages.append,
            session_id_factory=iter(("one", "two")).__next__,
            max_frame_bytes=4,
        )

    async def negotiated(self, client_id="client-1"):
        from xiaozhi_companion.loopback import LoopbackTransport

        transport = LoopbackTransport()
        session = await self.server.open_session(
            transport,
            {
                "Authorization": "Bearer test-token",
                "Protocol-Version": "1",
                "Device-Id": "device-1",
                "Client-Id": client_id,
            },
        )
        await session.receive_text(
            json.dumps(
                {
                    "type": "hello",
                    "version": 1,
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
        return session, transport

    @staticmethod
    def command(session, message_type, **fields):
        return json.dumps(
            {"session_id": session.session_id, "type": message_type, **fields}
        )

    async def test_abort_is_idempotent_and_never_emits_partial_voice(self):
        session, _transport = await self.negotiated()
        await session.receive_text(
            self.command(session, "listen", state="start", mode="manual")
        )
        generation = session.interaction_generation
        await session.receive_binary(b"opus")

        await session.receive_text(self.command(session, "abort"))
        await session.receive_text(self.command(session, "abort"))

        self.assertEqual(self.voice_messages, [])
        self.assertFalse(session.interaction_active)
        self.assertEqual(session.interaction_generation, generation + 1)

    async def test_abort_invalidates_late_outbound_but_next_turn_recovers(self):
        from xiaozhi_companion.protocol import ProtocolError

        session, transport = await self.negotiated()
        await session.receive_text(
            self.command(session, "listen", state="start", mode="manual")
        )
        old_generation = session.interaction_generation
        await session.receive_text(self.command(session, "abort"))

        with self.assertRaisesRegex(ProtocolError, "stale_interaction"):
            await session.send_transcript("private late text", generation=old_generation)
        stale_calls = (
            session.send_emotion("neutral", generation=old_generation),
            session.start_tts(generation=old_generation),
            session.send_tts_sentence("late", generation=old_generation),
            session.send_tts_audio(b"late", generation=old_generation),
            session.stop_tts(generation=old_generation),
        )
        for call in stale_calls:
            with self.assertRaisesRegex(ProtocolError, "stale_interaction"):
                await call
        self.assertEqual(len(transport.json_messages), 1)

        await session.receive_text(
            self.command(session, "listen", state="start", mode="manual")
        )
        await session.receive_binary(b"new")
        await session.receive_text(self.command(session, "listen", state="stop"))
        await session.wait_for_voice_processing()
        self.assertEqual(len(self.voice_messages), 1)
        self.assertEqual(self.voice_messages[0].frames, (b"new",))
        await session.send_transcript(
            "new request", generation=session.interaction_generation
        )
        self.assertEqual(transport.json_messages[-1]["text"], "new request")

    async def test_zero_and_oversized_audio_fail_closed(self):
        from xiaozhi_companion.protocol import ProtocolError

        for payload, expected in ((b"", "invalid_audio_frame"), (b"12345", "audio_frame_too_large")):
            session, _transport = await self.negotiated(
                client_id=f"client-{expected}"
            )
            await session.receive_text(
                self.command(session, "listen", state="start", mode="manual")
            )
            with self.assertRaisesRegex(ProtocolError, expected):
                await session.receive_binary(payload)


