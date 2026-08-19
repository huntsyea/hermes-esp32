import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PLUGIN_PARENT = ROOT / "tools" / "hermes_plugin"
HERMES_ROOT = Path(
    os.environ.get("HERMES_AGENT_ROOT", Path.home() / ".hermes" / "hermes-agent")
)
HERMES_PYTHON = HERMES_ROOT / "venv" / "bin" / "python"

sys.path.insert(0, str(PLUGIN_PARENT))


class XiaozhiProtocolContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from xiaozhi_companion.protocol import XiaozhiProtocolServer

        self.voice_messages = []
        self.diagnostics = []
        session_ids = iter(("session-1", "session-2", "session-3"))
        self.server = XiaozhiProtocolServer(
            expected_token="Bearer test-token",
            on_voice_message=self.voice_messages.append,
            diagnostic_sink=self.diagnostics.append,
            session_id_factory=lambda: next(session_ids),
        )

    @staticmethod
    def headers(
        *,
        token="Bearer test-token",
        protocol_version="1",
        device_id="aa:bb:cc:dd:ee:ff",
        client_id="client-1",
    ):
        return {
            "Authorization": token,
            "Protocol-Version": protocol_version,
            "Device-Id": device_id,
            "Client-Id": client_id,
        }

    @staticmethod
    def device_hello(**overrides):
        message = {
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
        message.update(overrides)
        return message

    async def negotiated_session(self, **header_overrides):
        from xiaozhi_companion.harness import ProtocolHarness

        harness = ProtocolHarness(self.server, self.headers(**header_overrides))
        session = await harness.connect()
        await harness.negotiate()
        return session, harness.transport

    async def test_hello_negotiates_required_v1_audio_parameters(self):
        session, transport = await self.negotiated_session()

        self.assertEqual(session.session_id, "session-1")
        self.assertEqual(
            transport.json_messages,
            [
                {
                    "type": "hello",
                    "version": 1,
                    "transport": "websocket",
                    "session_id": "session-1",
                    "audio_params": {
                        "format": "opus",
                        "sample_rate": 16000,
                        "channels": 1,
                        "frame_duration": 60,
                    },
                }
            ],
        )

    async def test_authentication_and_unsupported_versions_are_rejected(self):
        from xiaozhi_companion.loopback import LoopbackTransport
        from xiaozhi_companion.protocol import AuthenticationError, ProtocolError

        with self.assertRaises(AuthenticationError):
            await self.server.open_session(
                LoopbackTransport(), self.headers(token="Bearer wrong-secret")
            )

        with self.assertRaises(ProtocolError) as error:
            await self.server.open_session(
                LoopbackTransport(), self.headers(protocol_version="2")
            )
        self.assertEqual(error.exception.code, "unsupported_version")

    async def test_unsupported_hello_transport_and_audio_are_rejected(self):
        from xiaozhi_companion.loopback import LoopbackTransport
        from xiaozhi_companion.protocol import ProtocolError

        cases = (
            self.device_hello(transport="mqtt"),
            self.device_hello(
                audio_params={
                    "format": "opus",
                    "sample_rate": 24000,
                    "channels": 1,
                    "frame_duration": 60,
                }
            ),
        )
        expected_codes = ("unsupported_transport", "unsupported_audio_parameters")
        for index, (message, expected_code) in enumerate(
            zip(cases, expected_codes), start=1
        ):
            session = await self.server.open_session(
                LoopbackTransport(),
                self.headers(client_id=f"client-unsupported-{index}"),
            )
            with self.assertRaises(ProtocolError) as error:
                await session.receive_text(json.dumps(message))
            self.assertEqual(error.exception.code, expected_code)

        session = await self.server.open_session(
            LoopbackTransport(), self.headers(client_id="client-2")
        )
        with self.assertRaises(ProtocolError) as error:
            await session.receive_text(json.dumps(self.device_hello(version=2)))
        self.assertEqual(error.exception.code, "unsupported_version")

    async def test_listen_audio_stop_emits_one_voice_message(self):
        from xiaozhi_companion.harness import ProtocolHarness
        from xiaozhi_companion.protocol import ProtocolError

        harness = ProtocolHarness(self.server, self.headers())
        session = await harness.connect()
        await harness.negotiate()
        frames = (b"\xf8\xff\xfe", b"\x11\x22\x33\x44")

        await harness.start_listening()
        await harness.send_audio(*frames)
        await harness.stop_listening()

        self.assertEqual(len(self.voice_messages), 1)
        recording = self.voice_messages[0]
        self.assertEqual(recording.session_id, "session-1")
        self.assertEqual(recording.frames, frames)
        self.assertEqual(recording.audio_params.sample_rate, 16000)

        with self.assertRaisesRegex(ProtocolError, "listen_not_active"):
            await session.receive_text(
                json.dumps(
                    {
                        "session_id": session.session_id,
                        "type": "listen",
                        "state": "stop",
                    }
                )
            )
        self.assertEqual(len(self.voice_messages), 1)

    async def test_outbound_lifecycle_routes_only_to_active_session(self):
        from xiaozhi_companion.protocol import ProtocolError

        session, transport = await self.negotiated_session()

        await session.send_transcript("hello world")
        await session.send_emotion("happy", text="Hello")
        await session.start_tts()
        await session.send_tts_sentence("Hello")
        await session.send_tts_audio(b"opus-downlink")
        await session.stop_tts()

        self.assertEqual(
            transport.json_messages[1:],
            [
                {"session_id": "session-1", "type": "stt", "text": "hello world"},
                {
                    "session_id": "session-1",
                    "type": "llm",
                    "emotion": "happy",
                    "text": "Hello",
                },
                {"session_id": "session-1", "type": "tts", "state": "start"},
                {
                    "session_id": "session-1",
                    "type": "tts",
                    "state": "sentence_start",
                    "text": "Hello",
                },
                {"session_id": "session-1", "type": "tts", "state": "stop"},
            ],
        )
        self.assertEqual(transport.binary_messages, [b"opus-downlink"])

        replacement, replacement_transport = await self.negotiated_session(
            client_id="client-2"
        )
        self.assertFalse(session.is_active)
        self.assertTrue(replacement.is_active)
        self.assertEqual(transport.close_events[-1].code, 4001)

        with self.assertRaisesRegex(ProtocolError, "stale_session"):
            await session.send_tts_audio(b"must-not-cross")
        self.assertEqual(transport.binary_messages, [b"opus-downlink"])
        self.assertEqual(replacement_transport.binary_messages, [])

    async def test_malformed_unexpected_and_out_of_order_frames_fail_closed(self):
        from xiaozhi_companion.loopback import LoopbackTransport
        from xiaozhi_companion.protocol import ProtocolError

        session = await self.server.open_session(LoopbackTransport(), self.headers())
        cases = ("{", "[]", '"text"')
        for payload in cases:
            with self.assertRaises(ProtocolError):
                await session.receive_text(payload)

        session, _transport = await self.negotiated_session(client_id="client-2")
        with self.assertRaises(ProtocolError) as error:
            await session.receive_binary(b"unexpected")
        self.assertEqual(error.exception.code, "listen_not_active")

        with self.assertRaises(ProtocolError) as error:
            await session.receive_text(
                json.dumps(
                    {
                        "session_id": "stale-id",
                        "type": "listen",
                        "state": "start",
                        "mode": "manual",
                    }
                )
            )
        self.assertEqual(error.exception.code, "session_mismatch")

    async def test_clean_disconnect_discards_partial_audio(self):
        session, transport = await self.negotiated_session()
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
        await session.receive_binary(b"partial-audio")

        await session.disconnect()

        self.assertFalse(session.is_active)
        self.assertEqual(self.voice_messages, [])
        self.assertEqual(transport.close_events, [])

    async def test_diagnostics_never_include_tokens_or_audio_content(self):
        from xiaozhi_companion.loopback import LoopbackTransport
        from xiaozhi_companion.protocol import AuthenticationError, ProtocolError

        secret = "never-log-this-token"
        audio = b"never-log-these-audio-bytes"
        with self.assertRaises(AuthenticationError):
            await self.server.open_session(
                LoopbackTransport(), self.headers(token=f"Bearer {secret}")
            )

        session, _transport = await self.negotiated_session(client_id="client-2")
        with self.assertRaises(ProtocolError):
            await session.receive_binary(audio)

        rendered = repr(self.diagnostics)
        self.assertNotIn(secret, rendered)
        self.assertNotIn(audio.decode(), rendered)
        self.assertNotIn(audio.hex(), rendered)

    async def test_audio_buffer_limits_fail_closed(self):
        from xiaozhi_companion.harness import ProtocolHarness
        from xiaozhi_companion.protocol import ProtocolError, XiaozhiProtocolServer

        server = XiaozhiProtocolServer(
            expected_token="Bearer test-token",
            on_voice_message=self.voice_messages.append,
            session_id_factory=lambda: "bounded-session",
            max_frame_bytes=4,
            max_recording_bytes=6,
        )
        harness = ProtocolHarness(server, self.headers())
        await harness.connect()
        await harness.negotiate()
        await harness.start_listening()
        await harness.send_audio(b"1234")

        with self.assertRaisesRegex(ProtocolError, "recording_too_large"):
            await harness.send_audio(b"567")


@unittest.skipUnless(
    HERMES_PYTHON.is_file()
    and (HERMES_ROOT / "gateway" / "platforms" / "base.py").is_file(),
    "live Hermes checkout and venv are not available",
)
class XiaozhiHermesPluginContractTests(unittest.TestCase):
    def test_plugin_load_and_voice_boundary_use_live_public_hermes_api(self):
        script = r'''
import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.platforms.base import BasePlatformAdapter

from xiaozhi_companion.loopback import LoopbackTransport


class Context:
    def __init__(self):
        self.entry = None

    def register_platform(self, **kwargs):
        self.entry = PlatformEntry(source="plugin", **kwargs)
        platform_registry.register(self.entry)


async def main():
    module = importlib.import_module("xiaozhi_companion")
    context = Context()
    module.register(context)
    assert context.entry.name == "xiaozhi_companion"
    assert context.entry.check_fn()
    assert context.entry.allowed_users_env == "XIAOZHI_COMPANION_ALLOWED_DEVICES"
    assert context.entry.allow_all_env == "XIAOZHI_COMPANION_ALLOW_ALL_DEVICES"

    config = SimpleNamespace(extra={
        "token": "test-token",
        "transport": "loopback",
    })
    assert context.entry.validate_config(config)
    assert not context.entry.validate_config(
        SimpleNamespace(extra={"token": "test-token"})
    )
    adapter = context.entry.adapter_factory(
        config
    )
    assert isinstance(adapter, BasePlatformAdapter)
    events = []

    async def capture(event):
        events.append(event)

    adapter.handle_message = capture
    assert await adapter.connect()
    transport = LoopbackTransport()
    session = await adapter.protocol.open_session(
        transport,
        {
            "Authorization": "Bearer test-token",
            "Protocol-Version": "1",
            "Device-Id": "aa:bb:cc:dd:ee:ff",
            "Client-Id": "client-1",
        },
    )
    await session.receive_text(json.dumps({
        "type": "hello",
        "version": 1,
        "transport": "websocket",
        "audio_params": {
            "format": "opus",
            "sample_rate": 16000,
            "channels": 1,
            "frame_duration": 60,
        },
    }))
    send_result = await adapter.send("aa:bb:cc:dd:ee:ff", "test transcript")
    assert send_result.success
    assert transport.json_messages[-1] == {
        "session_id": session.session_id,
        "type": "stt",
        "text": "test transcript",
    }
    missing_result = await adapter.send("not-active", "must not route")
    assert not missing_result.success
    await session.receive_text(json.dumps({
        "session_id": session.session_id,
        "type": "listen",
        "state": "start",
        "mode": "manual",
    }))
    await session.receive_binary(b"\xf8\xff\xfe")
    await session.receive_text(json.dumps({
        "session_id": session.session_id,
        "type": "listen",
        "state": "stop",
    }))
    assert len(events) == 1
    assert events[0].message_type.value == "voice"
    assert events[0].source.chat_id == "aa:bb:cc:dd:ee:ff"
    assert events[0].metadata["xiaozhi_session_id"] == session.session_id
    assert events[0].raw_message.frames == (b"\xf8\xff\xfe",)
    assert events[0].media_types == ["audio/ogg"]
    assert len(events[0].media_urls) == 1
    assert Path(events[0].media_urls[0]).read_bytes().startswith(b"OggS")
    await adapter.disconnect()
    print(json.dumps({
        "registered": context.entry.name,
        "base_adapter": True,
        "voice_events": len(events),
    }))


asyncio.run(main())
'''
        with tempfile.TemporaryDirectory() as hermes_home:
            environment = os.environ.copy()
            environment["HERMES_HOME"] = hermes_home
            environment["PYTHONPATH"] = os.pathsep.join(
                (str(PLUGIN_PARENT), str(HERMES_ROOT))
            )
            completed = subprocess.run(
                [str(HERMES_PYTHON), "-c", script],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"{completed.stdout}\n{completed.stderr}",
        )
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(
            result,
            {
                "registered": "xiaozhi_companion",
                "base_adapter": True,
                "voice_events": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
