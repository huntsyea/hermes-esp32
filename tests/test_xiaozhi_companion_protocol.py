import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_PARENT = ROOT
HERMES_ROOT = Path(
    os.environ.get("HERMES_AGENT_ROOT", Path.home() / ".hermes" / "hermes-agent")
)
HERMES_PYTHON = HERMES_ROOT / "venv" / "bin" / "python"

sys.path.insert(0, str(PLUGIN_PARENT))


_END_OF_STREAM = object()


class QueueWebsocketConnection:
    def __init__(self, headers, *, path="/xiaozhi/v1"):
        self.request = SimpleNamespace(path=path, headers=headers)
        self.incoming = asyncio.Queue()
        self.sent = []
        self.closed = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = await self.incoming.get()
        if message is _END_OF_STREAM:
            raise StopAsyncIteration
        return message

    async def send(self, payload):
        self.sent.append(payload)

    async def close(self, *, code, reason):
        self.closed.append((code, reason))

    async def emit(self, payload):
        await self.incoming.put(payload)

    async def finish(self):
        await self.incoming.put(_END_OF_STREAM)


def device_hello_payload():
    return json.dumps(
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


def session_command(session_id, message_type, **fields):
    return json.dumps({"session_id": session_id, "type": message_type, **fields})


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
        await session.wait_for_voice_processing()

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


class WebsocketListenerVoiceLifecycleTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def headers(*, client_id="client-1"):
        return {
            "Authorization": "Bearer test-token",
            "Protocol-Version": "1",
            "Device-Id": "device-1",
            "Client-Id": client_id,
        }

    async def _start_listener_session(self, handler, *, diagnostics=None):
        from xiaozhi_companion.network import ListenerConfig, WebsocketListener
        from xiaozhi_companion.protocol import XiaozhiProtocolServer

        session_ids = iter(("session-1", "session-2"))
        server = XiaozhiProtocolServer(
            expected_token="Bearer test-token",
            on_voice_message=handler,
            diagnostic_sink=(diagnostics.append if diagnostics is not None else None),
            session_id_factory=lambda: next(session_ids),
        )
        listener = WebsocketListener(
            ListenerConfig(
                transport="loopback",
                bind_host="127.0.0.1",
                port=0,
                path="/xiaozhi/v1",
            ),
            server,
        )
        connection = QueueWebsocketConnection(self.headers())
        listener_task = asyncio.create_task(listener._handle_connection(connection))
        await connection.emit(device_hello_payload())
        for _ in range(100):
            session = server.active_session("device-1")
            if session is not None and session.is_negotiated:
                return server, session, connection, listener_task
            await asyncio.sleep(0)
        self.fail("listener session did not negotiate")

    async def test_abort_cancels_processing_and_listener_accepts_next_turn(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()
        next_turn_processed = asyncio.Event()
        release = asyncio.Event()
        messages = []
        cancellation_errors = []

        async def handle_voice(message):
            messages.append(message)
            if len(messages) == 1:
                started.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    from xiaozhi_companion.protocol import ProtocolError

                    try:
                        await session.send_transcript(
                            "late-private",
                            generation=message.interaction_generation,
                        )
                    except ProtocolError as error:
                        cancellation_errors.append(error.code)
                    cancelled.set()
                    raise
            else:
                next_turn_processed.set()

        server, session, connection, listener_task = await self._start_listener_session(
            handle_voice
        )
        await connection.emit(
            session_command(session.session_id, "listen", state="start", mode="manual")
        )
        await connection.emit(b"first-turn")
        await connection.emit(
            session_command(session.session_id, "listen", state="stop")
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        self.assertFalse(listener_task.done())

        await connection.emit(session_command(session.session_id, "abort"))
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        self.assertEqual(cancellation_errors, ["stale_interaction"])
        self.assertNotIn("late-private", repr(connection.sent))
        self.assertEqual(connection.closed, [])

        await connection.emit(
            session_command(session.session_id, "listen", state="start", mode="manual")
        )
        await connection.emit(b"next-turn")
        await connection.emit(
            session_command(session.session_id, "listen", state="stop")
        )
        await asyncio.wait_for(next_turn_processed.wait(), timeout=1)
        self.assertEqual(
            [message.frames for message in messages],
            [(b"first-turn",), (b"next-turn",)],
        )
        self.assertIs(server.active_session("device-1"), session)

        await connection.finish()
        await asyncio.wait_for(listener_task, timeout=1)

    async def test_disconnect_cancels_and_consumes_processing_task(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def handle_voice(_message):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        server, session, connection, listener_task = await self._start_listener_session(
            handle_voice
        )
        await connection.emit(
            session_command(session.session_id, "listen", state="start", mode="manual")
        )
        await connection.emit(b"disconnect-turn")
        await connection.emit(
            session_command(session.session_id, "listen", state="stop")
        )
        await asyncio.wait_for(started.wait(), timeout=1)

        await connection.finish()
        await asyncio.wait_for(listener_task, timeout=1)
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        self.assertIsNone(server.active_session("device-1"))
        self.assertEqual(connection.closed, [])

    async def test_session_replacement_cancels_processing_before_old_close(self):
        from xiaozhi_companion.loopback import LoopbackTransport

        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def handle_voice(_message):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        server, session, connection, listener_task = await self._start_listener_session(
            handle_voice
        )
        await connection.emit(
            session_command(session.session_id, "listen", state="start", mode="manual")
        )
        await connection.emit(b"replaced-turn")
        await connection.emit(
            session_command(session.session_id, "listen", state="stop")
        )
        await asyncio.wait_for(started.wait(), timeout=1)

        replacement = await server.open_session(
            LoopbackTransport(), self.headers(client_id="client-2")
        )
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        self.assertEqual(connection.closed, [(4001, "replaced")])
        self.assertIs(server.active_session("device-1"), replacement)

        await connection.finish()
        await asyncio.wait_for(listener_task, timeout=1)
        await replacement.disconnect()

    async def test_empty_and_rejected_modes_keep_transport_usable_for_manual_ptt(self):
        processed = asyncio.Queue()
        diagnostics = []

        async def handle_voice(message):
            await processed.put(message)

        server, session, connection, listener_task = await self._start_listener_session(
            handle_voice, diagnostics=diagnostics
        )

        await connection.emit(
            session_command(session.session_id, "listen", state="start", mode="manual")
        )
        for _ in range(100):
            if session.interaction_active:
                break
            await asyncio.sleep(0)
        first_generation = session.interaction_generation
        await connection.emit(
            session_command(session.session_id, "listen", state="stop")
        )
        for _ in range(100):
            if any(diagnostic.code == "empty_recording" for diagnostic in diagnostics):
                break
            await asyncio.sleep(0)
        else:
            self.fail("empty recording outcome was not observed")

        self.assertTrue(session.is_active)
        self.assertFalse(session.interaction_active)
        self.assertGreater(session.interaction_generation, first_generation)
        self.assertTrue(processed.empty())
        self.assertEqual(connection.closed, [])

        for mode in ("auto", "realtime"):
            await connection.emit(
                session_command(session.session_id, "listen", state="start", mode=mode)
            )
            await connection.emit(mode.encode("ascii"))
            await connection.emit(
                session_command(session.session_id, "listen", state="stop")
            )
            for _ in range(100):
                rejected = [
                    diagnostic
                    for diagnostic in diagnostics
                    if diagnostic.code == "listen_mode_rejected"
                ]
                if len(rejected) == (1 if mode == "auto" else 2):
                    break
                await asyncio.sleep(0)
            else:
                self.fail(f"{mode} rejection was not observed")
            self.assertTrue(processed.empty())
            self.assertIs(server.active_session("device-1"), session)
            self.assertEqual(connection.closed, [])

        await connection.emit(
            session_command(session.session_id, "listen", state="start", mode="manual")
        )
        await connection.emit(b"manual-next-turn")
        await connection.emit(
            session_command(session.session_id, "listen", state="stop")
        )
        message = await asyncio.wait_for(processed.get(), timeout=1)
        self.assertEqual(message.frames, (b"manual-next-turn",))
        error_messages = [
            json.loads(payload)
            for payload in connection.sent
            if isinstance(payload, str) and json.loads(payload).get("type") == "error"
        ]
        self.assertEqual([message["code"] for message in error_messages], ["agent", "agent"])
        self.assertEqual(connection.closed, [])

        await connection.finish()
        await asyncio.wait_for(listener_task, timeout=1)


