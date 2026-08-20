import asyncio
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PLUGIN_PARENT = ROOT / "tools" / "hermes_plugin"
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


class FirmwareRecoveryContractTests(unittest.TestCase):
    def test_session_capture_cancel_drain_and_downlink_gates_are_executable(self):
        source = textwrap.dedent(
            """
            #include <cassert>
            #include <string>
            #include "main/audio_downlink_gate.h"
            #include "main/capture_drain_gate.h"
            #include "main/conversation_session.h"
            #include "main/conversation_recovery_policy.h"
            #include "main/protocols/incoming_frame_limits.h"

            int main() {
                ConversationSession session;
                session.Open("session-1");
                assert(!session.IsActive());
                const auto first = session.StartInteraction();
                uint32_t captured = 0;
                assert(session.Capture("session-1", captured));
                assert(captured == first);
                std::string current_session;
                uint32_t current_epoch = 0;
                assert(session.CaptureCurrent(current_session, current_epoch));
                assert(current_session == "session-1" && current_epoch == first);

                CaptureDrainGate drain;
                drain.Begin(first);
                assert(!drain.Ready(first, false));
                assert(drain.Ready(first, true));
                drain.Cancel();
                assert(!drain.Ready(first, true));

                AudioDownlinkGate downlink;
                assert(!downlink.Accept(true));
                downlink.Start();
                assert(downlink.Accept(session.Matches("session-1", first)));
                assert(!downlink.Accept(false));
                downlink.Stop();
                assert(!downlink.Accept(true));

                assert(session.Cancel());
                const auto cancelled = session.epoch();
                assert(!session.Cancel());
                assert(session.epoch() == cancelled);
                assert(!session.Matches("session-1", first));

                const auto second = session.StartInteraction();
                assert(second != first);
                assert(session.Matches("session-1", second));
                session.Close();
                assert(!session.IsActive());
                assert(!session.Matches("session-1", second));

                assert(ShouldCloseConversationOnNetworkLoss(true, false, false, false));
                assert(ShouldCloseConversationOnNetworkLoss(false, true, false, false));
                assert(ShouldCloseConversationOnNetworkLoss(false, false, true, false));
                // Thinking uses the idle device state with an active conversation.
                assert(ShouldCloseConversationOnNetworkLoss(false, false, false, true));
                assert(!ShouldCloseConversationOnNetworkLoss(false, false, false, false));

                assert(IsValidIncomingPayload(1, 4));
                assert(!IsValidIncomingPayload(0, 4));
                assert(!IsValidIncomingPayload(5, 4));
                assert(IsValidIncomingFramedPayload(8, 4, 4, 4));
                assert(!IsValidIncomingFramedPayload(8, 4, 0, 4));
                assert(!IsValidIncomingFramedPayload(8, 4, 3, 4));
                assert(!IsValidIncomingFramedPayload(8, 4, 5, 4));
                return 0;
            }
            """
        )
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "recovery.cc"
            binary = Path(directory) / "recovery"
            source_path.write_text(source, encoding="utf-8")
            subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    "-pthread",
                    "-I",
                    str(ROOT),
                    str(source_path),
                    "-o",
                    str(binary),
                ],
                check=True,
            )
            subprocess.run([str(binary)], check=True)

    def test_application_routes_recovery_through_one_central_cleanup(self):
        header = (ROOT / "main" / "application.h").read_text(encoding="utf-8")
        source = (ROOT / "main" / "application.cc").read_text(encoding="utf-8")

        self.assertIn("CancelConversation", header)
        self.assertIn("DiscardPendingCapture", source)
        self.assertIn("IncomingSessionMatches", source)
        self.assertIn("ConversationSession conversation_session_", header)
        self.assertIn("RequiresConversationSessionId", source)
        self.assertIn("capture_stop_.Begin", source)
        self.assertIn("FinishListeningAfterCaptureDrain", source)
        self.assertIn("ShouldCloseConversationOnNetworkLoss", source)
        self.assertIn("protocol_->CloseAudioChannel(false)", source)
        connecting_stop = source.split("void Application::HandleStopListeningEvent", 1)[1].split(
            "void Application::HandleWakeWordDetectedEvent", 1
        )[0]
        self.assertIn("protocol_->CloseAudioChannel(false)", connecting_stop)

    def test_protocol_callbacks_schedule_application_mutation(self):
        source = (ROOT / "main" / "application.cc").read_text(encoding="utf-8")

        closed = source.split("protocol_->OnAudioChannelClosed", 1)[1].split(
            "protocol_->OnIncomingJson", 1
        )[0]
        self.assertIn("Schedule([this]", closed)
        schedule_body = closed.split("Schedule([this]", 1)[1]
        self.assertIn("SetPowerSaveLevel", schedule_body)
        self.assertIn("conversation_session_.Close()", schedule_body)
        self.assertIn("tts_downlink_.Stop()", schedule_body)

    def test_websocket_rejects_malformed_audio_bounds_and_tags_session(self):
        protocol = (ROOT / "main" / "protocols" / "websocket_protocol.cc").read_text(
            encoding="utf-8"
        )
        packet = (ROOT / "main" / "protocols" / "protocol.h").read_text(
            encoding="utf-8"
        )

        self.assertIn("kMaxIncomingAudioFrameBytes", protocol)
        self.assertIn("IsValidIncomingFramedPayload", protocol)
        self.assertIn("session_id = session_id_", protocol)
        self.assertIn("std::string session_id", packet)

    def test_cancel_discards_capture_and_bounds_active_output_to_one_frame(self):
        header = (ROOT / "main" / "audio" / "audio_service.h").read_text(
            encoding="utf-8"
        )
        source = (ROOT / "main" / "audio" / "audio_service.cc").read_text(
            encoding="utf-8"
        )

        self.assertIn("DiscardPendingCapture", header)
        self.assertIn("capture_generation_", header)
        self.assertIn("task->generation == capture_generation_", source)
        self.assertIn("one codec OutputData call", source)

    def test_private_content_is_not_written_to_firmware_logs(self):
        source = (ROOT / "main" / "application.cc").read_text(encoding="utf-8")
        forbidden = ('ESP_LOGI(TAG, ">> %s"', 'ESP_LOGI(TAG, "<< %s"')
        for value in forbidden:
            self.assertNotIn(value, source)


@unittest.skipUnless(
    HERMES_PYTHON.is_file()
    and (HERMES_ROOT / "gateway" / "platforms" / "base.py").is_file(),
    "live Hermes checkout and venv are not available",
)
class LiveHermesRecoveryTests(unittest.TestCase):
    def test_cancelled_provider_and_service_failures_are_safe_and_recoverable(self):
        script = r'''
import asyncio
import json
from types import SimpleNamespace

from xiaozhi_companion.loopback import LoopbackTransport
from gateway.platform_registry import PlatformEntry, platform_registry


class Context:
    def register_platform(self, **kwargs):
        platform_registry.register(PlatformEntry(source="plugin", **kwargs))


def command(session, message_type, **fields):
    return json.dumps({"session_id": session.session_id, "type": message_type, **fields})


async def record(session, payload=b"opus"):
    await session.receive_text(command(session, "listen", state="start", mode="manual"))
    await session.receive_binary(payload)
    await session.receive_text(command(session, "listen", state="stop"))
    await session.wait_for_voice_processing()


async def main():
    import xiaozhi_companion
    xiaozhi_companion.register(Context())
    from xiaozhi_companion.adapter import XiaozhiCompanionAdapter, VoiceResponseContext

    adapter = XiaozhiCompanionAdapter(
        SimpleNamespace(extra={"token": "test-token", "transport": "loopback"}),
        session_id_factory=lambda: "session-1",
        ogg_serial_factory=lambda: 7,
        audio_cache=lambda *_args, **_kwargs: "/tmp/fixture.ogg",
        tts_audio_loader=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError()),
    )
    transport = LoopbackTransport()
    session = await adapter.protocol.open_session(
        transport,
        {
            "Authorization": "Bearer test-token",
            "Protocol-Version": "1",
            "Device-Id": "device-1",
            "Client-Id": "client-1",
        },
    )
    await session.receive_text(json.dumps({
        "type": "hello", "version": 1, "transport": "websocket",
        "audio_params": {"format": "opus", "sample_rate": 16000,
                         "channels": 1, "frame_duration": 60},
    }))

    provider_started = asyncio.Event()
    provider_cancelled = asyncio.Event()
    provider_release = asyncio.Event()
    provider_results = []

    async def delayed_provider(_event):
        provider_started.set()
        try:
            await provider_release.wait()
            provider_results.append(await adapter.send(
                "device-1", "private late response", metadata={"notify": True}
            ))
        except asyncio.CancelledError:
            provider_cancelled.set()
            raise

    adapter.handle_message = delayed_provider
    await session.receive_text(command(session, "listen", state="start", mode="manual"))
    await session.receive_binary(b"old")
    stop_task = asyncio.create_task(
        session.receive_text(command(session, "listen", state="stop"))
    )
    await provider_started.wait()
    await session.receive_text(command(session, "abort"))
    await asyncio.wait_for(provider_cancelled.wait(), timeout=1)
    provider_release.set()
    await stop_task
    assert not provider_results
    assert not any("private late response" in json.dumps(item) for item in transport.json_messages)

    async def failing_agent(_event):
        raise RuntimeError("private provider detail")

    adapter.handle_message = failing_agent
    await record(session, b"agent")
    assert transport.json_messages[-1]["type"] == "error"
    assert transport.json_messages[-1]["code"] == "agent"

    async def successful_agent(_event):
        return None

    adapter.handle_message = successful_agent
    await session.receive_text(command(session, "listen", state="start", mode="manual"))
    generation = session.interaction_generation
    await session.receive_binary(b"tts")
    await session.receive_text(command(session, "listen", state="stop"))
    await session.wait_for_voice_processing()
    token = adapter._voice_response_context.set(
        VoiceResponseContext(session, interaction_generation=generation)
    )
    try:
        adapter.prepare_tts_text("private speech")
        result = await adapter.play_tts("device-1", "/not/read")
        assert not result.success
    finally:
        adapter._voice_response_context.reset(token)
    assert transport.json_messages[-1]["code"] == "tts"

    adapter._audio_cache = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError())
    await record(session, b"stt")
    assert transport.json_messages[-1]["code"] == "stt"

    adapter._audio_cache = lambda *_args, **_kwargs: "/tmp/fixture.ogg"
    before = len(transport.json_messages)
    await record(session, b"recovered")
    assert len(transport.json_messages) == before
    assert session.interaction_active


asyncio.run(main())
'''
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(PLUGIN_PARENT), str(HERMES_ROOT), environment.get("PYTHONPATH", ""))
        )
        completed = subprocess.run(
            [str(HERMES_PYTHON), "-c", script],
            cwd=HERMES_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_abort_removes_cache_file_written_by_in_flight_worker(self):
        script = r'''
import asyncio
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

from gateway.platform_registry import PlatformEntry, platform_registry
from xiaozhi_companion.loopback import LoopbackTransport


class Context:
    def register_platform(self, **kwargs):
        platform_registry.register(PlatformEntry(source="plugin", **kwargs))


def command(session, message_type, **fields):
    return json.dumps({"session_id": session.session_id, "type": message_type, **fields})


async def main(cache_path):
    import xiaozhi_companion

    xiaozhi_companion.register(Context())
    from xiaozhi_companion.adapter import XiaozhiCompanionAdapter

    written = threading.Event()
    release = threading.Event()
    events = []

    def delayed_cache(data, *, ext):
        del ext
        Path(cache_path).write_bytes(data)
        written.set()
        release.wait(timeout=2)
        return cache_path

    adapter = XiaozhiCompanionAdapter(
        SimpleNamespace(extra={"token": "test-token", "transport": "loopback"}),
        session_id_factory=lambda: "cache-session",
        ogg_serial_factory=lambda: 11,
        audio_cache=delayed_cache,
    )

    async def capture(event):
        events.append(event)

    adapter.handle_message = capture
    transport = LoopbackTransport()
    session = await adapter.protocol.open_session(
        transport,
        {
            "Authorization": "Bearer test-token",
            "Protocol-Version": "1",
            "Device-Id": "device-1",
            "Client-Id": "client-1",
        },
    )
    await session.receive_text(json.dumps({
        "type": "hello", "version": 1, "transport": "websocket",
        "audio_params": {"format": "opus", "sample_rate": 16000,
                         "channels": 1, "frame_duration": 60},
    }))
    await session.receive_text(command(session, "listen", state="start", mode="manual"))
    await session.receive_binary(b"private-audio")
    await session.receive_text(command(session, "listen", state="stop"))
    assert await asyncio.to_thread(written.wait, 1)

    await session.receive_text(command(session, "abort"))
    release.set()
    for _ in range(100):
        if not Path(cache_path).exists():
            break
        await asyncio.sleep(0.01)
    assert not Path(cache_path).exists()
    assert events == []
    await adapter.disconnect()


asyncio.run(main(sys.argv[1]))
print("cancelled-cache-cleanup-ok")
'''
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cancelled-recording.ogg"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                (str(PLUGIN_PARENT), str(HERMES_ROOT), environment.get("PYTHONPATH", ""))
            )
            completed = subprocess.run(
                [str(HERMES_PYTHON), "-c", script, str(cache_path)],
                cwd=HERMES_ROOT,
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
        self.assertIn("cancelled-cache-cleanup-ok", completed.stdout)


if __name__ == "__main__":
    unittest.main()
