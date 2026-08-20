import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from scripts.tests.companion_test_support import (
        ensure_simulator,
        simulator_environment,
    )
except ModuleNotFoundError:  # firmware tree not part of hermes-esp32
    ensure_simulator = None
    simulator_environment = None


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


