import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.tests.companion_test_support import (
    ensure_simulator,
    simulator_environment,
)


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_PARENT = ROOT
HERMES_ROOT = Path(
    os.environ.get("HERMES_AGENT_ROOT", Path.home() / ".hermes" / "hermes-agent")
)
HERMES_PYTHON = HERMES_ROOT / "venv" / "bin" / "python"
SIMULATOR = ROOT / "build" / "companion-simulator-macos" / "companion-simulator"

sys.path.insert(0, str(PLUGIN_PARENT))


def ogg_crc(page):
    checksum_page = bytearray(page)
    checksum_page[22:26] = b"\0\0\0\0"
    crc = 0
    for byte in checksum_page:
        crc ^= byte << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if crc & 0x80000000 else (crc << 1) & 0xFFFFFFFF
    return crc


def parse_ogg_pages(payload):
    pages = []
    offset = 0
    while offset < len(payload):
        if payload[offset : offset + 4] != b"OggS":
            raise AssertionError(f"Missing Ogg capture pattern at {offset}")
        segment_count = payload[offset + 26]
        lacing = payload[offset + 27 : offset + 27 + segment_count]
        page_size = 27 + segment_count + sum(lacing)
        page = payload[offset : offset + page_size]
        pages.append(
            {
                "raw": page,
                "flags": page[5],
                "granule": struct.unpack_from("<Q", page, 6)[0],
                "serial": struct.unpack_from("<I", page, 14)[0],
                "sequence": struct.unpack_from("<I", page, 18)[0],
                "checksum": struct.unpack_from("<I", page, 22)[0],
                "body": page[27 + segment_count :],
            }
        )
        offset += page_size
    return pages


class OggOpusPacketizerTests(unittest.TestCase):
    def test_fixed_60_ms_packets_form_deterministic_valid_ogg_pages(self):
        from xiaozhi_companion.ogg_opus import packets_to_ogg_opus

        frames = (b"\xf8\xff\xfe", b"\xf8\xff\xfe")
        payload = packets_to_ogg_opus(frames, serial=0x1234ABCD)
        duplicate = packets_to_ogg_opus(frames, serial=0x1234ABCD)
        pages = parse_ogg_pages(payload)

        self.assertEqual(payload, duplicate)
        self.assertEqual([page["sequence"] for page in pages], list(range(4)))
        self.assertEqual({page["serial"] for page in pages}, {0x1234ABCD})
        self.assertEqual([page["granule"] for page in pages], [0, 0, 2880, 5760])
        self.assertEqual(pages[0]["flags"], 0x02)
        self.assertEqual(pages[-1]["flags"], 0x04)
        self.assertTrue(pages[0]["body"].startswith(b"OpusHead"))
        self.assertEqual(struct.unpack_from("<H", pages[0]["body"], 10)[0], 0)
        self.assertTrue(pages[1]["body"].startswith(b"OpusTags"))
        self.assertEqual([page["body"] for page in pages[2:]], list(frames))
        for page in pages:
            self.assertEqual(page["checksum"], ogg_crc(page["raw"]))

    def test_empty_oversized_and_excessive_packet_sets_fail_closed(self):
        from xiaozhi_companion.ogg_opus import OggOpusError, packets_to_ogg_opus

        cases = (
            (),
            (b"x" * (16 * 1024 + 1),),
            tuple(b"x" for _ in range(5001)),
        )
        for frames in cases:
            with self.subTest(frame_count=len(frames)):
                with self.assertRaises(OggOpusError):
                    packets_to_ogg_opus(frames, serial=1)


class PttProtocolBoundsTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def headers():
        return {
            "Authorization": "Bearer test-token",
            "Protocol-Version": "1",
            "Device-Id": "device-1",
            "Client-Id": "client-1",
        }

    async def test_empty_recording_is_recoverable_and_does_not_emit_voice(self):
        from xiaozhi_companion.harness import ProtocolHarness
        from xiaozhi_companion.protocol import XiaozhiProtocolServer

        messages = []
        server = XiaozhiProtocolServer(
            expected_token="Bearer test-token",
            on_voice_message=messages.append,
            session_id_factory=lambda: "empty-session",
        )
        harness = ProtocolHarness(server, self.headers())
        await harness.connect()
        await harness.negotiate()
        await harness.start_listening()
        generation = harness.session.interaction_generation
        await harness.stop_listening()
        self.assertEqual(messages, [])
        self.assertTrue(harness.session.is_active)
        self.assertFalse(harness.session.interaction_active)
        self.assertEqual(harness.session.interaction_generation, generation + 1)

        await harness.start_listening()
        await harness.send_audio(b"next-turn")
        await harness.stop_listening()
        await harness.session.wait_for_voice_processing()
        self.assertEqual([message.frames for message in messages], [(b"next-turn",)])

    async def test_recording_frame_count_is_bounded(self):
        from xiaozhi_companion.harness import ProtocolHarness
        from xiaozhi_companion.protocol import ProtocolError, XiaozhiProtocolServer

        server = XiaozhiProtocolServer(
            expected_token="Bearer test-token",
            on_voice_message=lambda _message: None,
            session_id_factory=lambda: "bounded-session",
            max_recording_frames=2,
        )
        harness = ProtocolHarness(server, self.headers())
        await harness.connect()
        await harness.negotiate()
        await harness.start_listening()
        await harness.send_audio(b"one", b"two")
        with self.assertRaisesRegex(ProtocolError, "too_many_audio_frames"):
            await harness.send_audio(b"three")


class CompanionPttUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_simulator()

    def run_simulator(self, *arguments):
        completed = subprocess.run(
            [str(SIMULATOR), *arguments],
            cwd=ROOT,
            env=simulator_environment(),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, msg=f"{completed.stdout}\n{completed.stderr}")
        return completed.stdout.strip()

    def test_transcript_is_centrally_bounded_and_clears_for_next_recording(self):
        transcript = "say " + ("something " * 30)
        fields = dict(
            field.split("=", 1)
            for field in self.run_simulator("--inspect-transcript", transcript).split()
        )
        self.assertEqual(fields["bytes"], "96")
        self.assertEqual(fields["truncated"], "yes")
        self.assertEqual(fields["cleared_on_listen"], "yes")

    def test_board_and_application_wire_press_hold_ptt_and_semantic_events(self):
        board = (
            ROOT
            / "main"
            / "boards"
            / "waveshare"
            / "esp32-s3-touch-lcd-1.54"
            / "esp32-s3-touch-lcd-1.54.cc"
        ).read_text(encoding="utf-8")
        application = (ROOT / "main" / "application.cc").read_text(encoding="utf-8")
        audio_header = (ROOT / "main" / "audio" / "audio_service.h").read_text(encoding="utf-8")

        self.assertIn("pwr_button_.OnPressDown", board)
        self.assertIn("app.StartListening()", board)
        self.assertIn("StopListening()", board)
        self.assertIn("SetPushToTalkActive(true)", board)
        self.assertIn("SetPushToTalkActive(false)", board)
        self.assertIn("PUSH_TO_TALK_HOLD_MS 250", board)
        self.assertGreaterEqual(
            board.count("push_to_talk_consumed_click_.exchange(false)"), 3
        )
        self.assertIn("pwr_button_.OnLongPress", board)
        self.assertIn("pwr_button_.OnDoubleClick", board)
        self.assertIn("pwr_button_.OnMultipleClick", board)
        self.assertIn("power_manager_->PowerOff()", board)
        for event in (
            "ConversationEvent::kListenStarted",
            "ConversationEvent::kRecordingEnded",
            "ConversationEvent::kTtsStarted",
            "ConversationEvent::kTtsStopped",
            "ConversationEvent::kTemporaryError",
            "ConversationEvent::kCancelled",
        ):
            self.assertIn(event, application)
        self.assertIn("bool EnableVoiceProcessing(bool enable)", audio_header)
        self.assertNotIn('ESP_LOGI(TAG, ">> %s"', application)
        self.assertNotIn('ESP_LOGI(TAG, "<< %s"', application)

    def test_board_queues_ui_edges_and_cancels_ptt_before_main_task_cleanup(self):
        board = (
            ROOT
            / "main"
            / "boards"
            / "waveshare"
            / "esp32-s3-touch-lcd-1.54"
            / "esp32-s3-touch-lcd-1.54.cc"
        ).read_text(encoding="utf-8")

        self.assertIn("kPendingConversationEventCapacity = 12", board)
        self.assertIn("std::array<product_ui_event_t, kPendingConversationEventCapacity>", board)
        self.assertNotIn("std::deque<PendingUiEvent>", board)
        self.assertIn("QueueConversationEvent(product_event)", board)
        self.assertIn("QueuePushToTalkEvent(active)", board)
        self.assertIn("FlushPendingUiEventsLocked();", board)
        self.assertIn("if (!clock_changed && !HasPendingUiEvents())", board)
        self.assertIn("ReconcilePendingConversationEvents(event)", board)
        self.assertIn("AppendPendingConversationEvent(PRODUCT_UI_EVENT_DISCONNECTED)", board)
        self.assertIn('ESP_LOGW(TAG, "UI event queue full;', board)
        self.assertIn('ESP_LOGE(TAG, "Failed to create Screen Manager")', board)
        self.assertIn("pending_push_to_talk_state_ = active", board)
        self.assertIn("has_pending_push_to_talk_state_ = true", board)

        release_start = board.index("void ReleasePushToTalk()")
        release_end = board.index("    void InitializePowerManager()", release_start)
        release_body = board[release_start:release_end]
        self.assertLess(
            release_body.index("Application::GetInstance().StopListening();"),
            release_body.index("Application::GetInstance().Schedule"),
        )


@unittest.skipUnless(
    HERMES_PYTHON.is_file()
    and (HERMES_ROOT / "gateway" / "platforms" / "base.py").is_file(),
    "live Hermes checkout and venv are not available",
)
class LiveHermesPttBoundaryTests(unittest.TestCase):
    def test_voice_event_caches_ogg_and_transcript_routes_to_same_session(self):
        script = r'''
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from xiaozhi_companion.loopback import LoopbackTransport
from gateway.platform_registry import PlatformEntry, platform_registry


class Context:
    def __init__(self):
        self.entry = None

    def register_platform(self, **kwargs):
        self.entry = PlatformEntry(source="plugin", **kwargs)
        platform_registry.register(self.entry)


async def main():
    import xiaozhi_companion

    context = Context()
    xiaozhi_companion.register(context)
    from xiaozhi_companion.adapter import XiaozhiCompanionAdapter

    adapter = XiaozhiCompanionAdapter(
        SimpleNamespace(extra={"token": "test-token", "transport": "loopback"}),
        session_id_factory=lambda: "ptt-session",
        ogg_serial_factory=lambda: 0x1234ABCD,
    )
    events = []
    event_received = asyncio.Event()

    async def capture(event):
        events.append(event)
        event_received.set()

    adapter.handle_message = capture
    await adapter.connect()
    transport = LoopbackTransport()
    session = await adapter.protocol.open_session(transport, {
        "Authorization": "Bearer test-token",
        "Protocol-Version": "1",
        "Device-Id": "device-1",
        "Client-Id": "client-1",
    })
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
    await asyncio.wait_for(event_received.wait(), timeout=1)
    await session.wait_for_voice_processing()

    assert len(events) == 1
    event = events[0]
    assert event.message_type.value == "voice"
    assert event.media_types == ["audio/ogg"]
    assert len(event.media_urls) == 1
    cached = Path(event.media_urls[0])
    assert cached.is_absolute() and cached.is_file()
    assert cached.read_bytes().startswith(b"OggS")
    assert event.source.chat_id == "device-1"
    assert event.metadata["xiaozhi_session_id"] == "ptt-session"

    result = await adapter.send("device-1", "recognized request")
    assert result.success
    assert transport.json_messages[-1] == {
        "session_id": "ptt-session",
        "type": "stt",
        "text": "recognized request",
    }
    await adapter.disconnect()
    print(json.dumps({"events": len(events), "mime": event.media_types[0]}))


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
        self.assertEqual(completed.returncode, 0, msg=f"{completed.stdout}\n{completed.stderr}")
        self.assertEqual(
            json.loads(completed.stdout.strip().splitlines()[-1]),
            {"events": 1, "mime": "audio/ogg"},
        )


if __name__ == "__main__":
    unittest.main()
