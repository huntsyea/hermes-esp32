import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PLUGIN_PARENT = ROOT / "tools" / "hermes_plugin"
HERMES_ROOT = Path(
    os.environ.get("HERMES_AGENT_ROOT", Path.home() / ".hermes" / "hermes-agent")
)
HERMES_PYTHON = HERMES_ROOT / "venv" / "bin" / "python"

sys.path.insert(0, str(PLUGIN_PARENT))


class TtsAudioTests(unittest.TestCase):
    def test_tts_ingress_publish_is_atomic_with_stop_completion(self):
        source = textwrap.dedent(
            r"""
            #include <atomic>
            #include <cassert>
            #include <condition_variable>
            #include <mutex>
            #include <thread>
            #include "main/audio_downlink_gate.h"

            int main() {
              AudioDownlinkGate gate;
              gate.Start();
              std::mutex ingress_mutex;
              std::mutex step_mutex;
              std::condition_variable step;
              std::atomic<bool> accepted{false};
              std::atomic<bool> stop_started{false};
              bool queued = false;
              bool finished = false;

              std::thread ingress([&] {
                std::lock_guard<std::mutex> lock(ingress_mutex);
                accepted = gate.Accept(true);
                {
                  std::lock_guard<std::mutex> step_lock(step_mutex);
                  step.notify_all();
                }
                while (!stop_started) {
                  std::this_thread::yield();
                }
                if (accepted) {
                  queued = true;
                }
              });
              std::thread stop([&] {
                std::unique_lock<std::mutex> step_lock(step_mutex);
                step.wait(step_lock, [&] { return accepted.load(); });
                gate.Stop();
                stop_started = true;
                step_lock.unlock();
                std::lock_guard<std::mutex> lock(ingress_mutex);
                finished = !queued;
              });

              ingress.join();
              stop.join();
              assert(accepted);
              assert(queued);
              assert(!finished);
              return 0;
            }
            """
        )
        with tempfile.TemporaryDirectory() as directory:
            cpp = Path(directory) / "tts_ingress_race.cc"
            binary = Path(directory) / "tts_ingress_race"
            cpp.write_text(source, encoding="utf-8")
            subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    "-pthread",
                    "-I",
                    str(ROOT),
                    str(cpp),
                    "-o",
                    str(binary),
                ],
                check=True,
            )
            subprocess.run([str(binary)], timeout=10, check=True)

    def test_response_preview_is_utf8_safe_and_bounded(self):
        from xiaozhi_companion.response_text import response_preview

        preview = response_preview("hello 🌊" * 80)
        self.assertLessEqual(len(preview.encode("utf-8")), 96)
        self.assertTrue(preview.endswith("..."))

    def test_demux_returns_only_ordered_opus_audio_packets(self):
        from xiaozhi_companion.ogg_opus import packets_to_ogg_opus
        from xiaozhi_companion.tts_audio import demux_ogg_opus

        frames = (b"\x18\x01", b"\x18\x02", b"\x18\x03")
        ogg = packets_to_ogg_opus(frames, serial=7)
        self.assertEqual(demux_ogg_opus(ogg), frames)

    def test_demux_rejects_malformed_wrong_parameters_and_bounds(self):
        from xiaozhi_companion.ogg_opus import packets_to_ogg_opus
        from xiaozhi_companion.tts_audio import TtsAudioError, demux_ogg_opus

        with self.assertRaises(TtsAudioError):
            demux_ogg_opus(b"not ogg")
        stereo = packets_to_ogg_opus((b"\x18\x01",), serial=8)
        stereo = stereo.replace(b"OpusHead\x01\x01", b"OpusHead\x01\x02", 1)
        with self.assertRaises(TtsAudioError):
            demux_ogg_opus(stereo, verify_checksum=False)
        too_many = packets_to_ogg_opus((b"\x18\x01", b"\x18\x02"), serial=9)
        with self.assertRaises(TtsAudioError):
            demux_ogg_opus(too_many, max_packets=1)

    def test_non_ogg_source_uses_safe_bounded_ffmpeg_argv(self):
        from xiaozhi_companion.ogg_opus import packets_to_ogg_opus
        from xiaozhi_companion.tts_audio import load_tts_opus_packets

        output = packets_to_ogg_opus((b"\x18\x01",), serial=10)
        calls = []

        def runner(argv, **kwargs):
            calls.append((argv, kwargs))
            return subprocess.CompletedProcess(argv, 0, output, b"")

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "audio with spaces.mp3"
            source.write_bytes(b"ID3 fixture")
            packets = load_tts_opus_packets(
                source, ffmpeg_executable="/safe/ffmpeg", runner=runner
            )

        self.assertEqual(packets, (b"\x18\x01",))
        argv, kwargs = calls[0]
        self.assertIsInstance(argv, list)
        self.assertEqual(argv[0], "/safe/ffmpeg")
        self.assertNotIn("shell", kwargs)
        self.assertIn("pipe:1", argv)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
    def test_real_ffmpeg_transcodes_pcm_to_negotiated_packets(self):
        from xiaozhi_companion.tts_audio import load_tts_opus_packets

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "silence.wav"
            with wave.open(str(source), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"\0\0" * (16000 * 180 // 1000))
            packets = load_tts_opus_packets(
                source, ffmpeg_executable=shutil.which("ffmpeg")
            )

        self.assertGreaterEqual(len(packets), 1)
        self.assertTrue(all(packet and not packet.startswith(b"Opus") for packet in packets))

    def test_source_and_transcode_output_are_bounded(self):
        from xiaozhi_companion.tts_audio import TtsAudioError, load_tts_opus_packets

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "large.ogg"
            source.write_bytes(b"OggS" + b"x" * 32)
            with self.assertRaises(TtsAudioError):
                load_tts_opus_packets(source, max_container_bytes=8)


class PlaybackEnvelopeTests(unittest.TestCase):
    def test_pcm_peak_envelope_is_bounded_for_silence_short_and_multiple_samples(self):
        source = textwrap.dedent(
            """
            #include <cassert>
            #include <cstdint>
            #include "main/audio/playback_envelope.h"
            int main() {
              assert(CalculatePlaybackEnvelope(nullptr, 0) == 0);
              const int16_t silence[] = {0, 0, 0};
              assert(CalculatePlaybackEnvelope(silence, 3) == 0);
              const int16_t short_audio[] = {-16384};
              auto half = CalculatePlaybackEnvelope(short_audio, 1);
              assert(half >= 499 && half <= 501);
              const int16_t full_audio[] = {1, -32768, 4, 5};
              assert(CalculatePlaybackEnvelope(full_audio, 4) == 1000);
              return 0;
            }
            """
        )
        with tempfile.TemporaryDirectory() as directory:
            cpp = Path(directory) / "test.cc"
            binary = Path(directory) / "test"
            cpp.write_text(source, encoding="utf-8")
            subprocess.run(
                ["c++", "-std=c++17", "-I", str(ROOT), str(cpp), "-o", str(binary)],
                check=True,
            )
            subprocess.run([str(binary)], check=True)

    def test_firmware_defers_tts_stop_until_playback_drains(self):
        application = (ROOT / "main" / "application.cc").read_text(encoding="utf-8")
        audio = (ROOT / "main" / "audio" / "audio_service.cc").read_text(
            encoding="utf-8"
        )
        display = (ROOT / "main" / "display" / "display.h").read_text(
            encoding="utf-8"
        )
        board = (
            ROOT
            / "main"
            / "boards"
            / "waveshare"
            / "esp32-s3-touch-lcd-1.54"
            / "esp32-s3-touch-lcd-1.54.cc"
        ).read_text(encoding="utf-8")
        self.assertIn("pending_tts_stop_", application)
        self.assertIn("FinishTtsPlayback", application)
        self.assertIn("MAIN_EVENT_PLAYBACK_DRAINED", application)
        self.assertIn("MAIN_EVENT_INCOMING_AUDIO", application)
        self.assertIn("pending_tts_start_epoch_", application)
        self.assertIn("HasPendingTtsPackets(epoch)", application)
        self.assertIn("tts_downlink_.Start()", application)
        self.assertIn("tts_downlink_.Accept", application)
        self.assertIn("kMaxPendingTtsPackets", application)
        self.assertIn("conversation_session_.Matches(pending.session_id, pending.epoch)", application)
        self.assertIn("PushPacketToDecodeQueue(std::move(pending.packet), false)", application)
        self.assertNotIn("PushPacketToDecodeQueue(std::move(packet), true)", application)
        ingress = application.split("protocol_->OnIncomingAudio", 1)[1].split(
            "protocol_->OnAudioChannelOpened", 1
        )[0]
        self.assertLess(
            ingress.index("std::lock_guard<std::mutex> lock(mutex_)"),
            ingress.index("tts_downlink_.Accept(true)"),
        )
        drain = application.split("void Application::DrainPendingTtsPackets", 1)[1].split(
            "void Application::ClearPendingTtsPackets", 1
        )[0]
        self.assertNotIn("tts_downlink_.Accept", drain)
        self.assertIn("on_playback_envelope", audio)
        self.assertIn("CalculatePlaybackEnvelope", audio)
        self.assertIn("SetPlaybackEnvelope", display)
        self.assertIn("product_ui_set_playback_envelope", board)
        self.assertNotIn('SetChatMessage("assistant", message.c_str())', application)


@unittest.skipUnless(
    HERMES_PYTHON.is_file()
    and (HERMES_ROOT / "gateway" / "platforms" / "base.py").is_file(),
    "live Hermes checkout and venv are not available",
)
class LiveHermesTtsBoundaryTests(unittest.TestCase):
    def test_one_response_is_not_duplicated_and_stale_session_is_rejected(self):
        script = r'''
import asyncio
import json
from types import SimpleNamespace

from xiaozhi_companion.loopback import LoopbackTransport
from gateway.platform_registry import PlatformEntry, platform_registry

class Context:
    def register_platform(self, **kwargs):
        platform_registry.register(PlatformEntry(source="plugin", **kwargs))

async def negotiated(adapter, client_id):
    transport = LoopbackTransport()
    session = await adapter.protocol.open_session(transport, {
        "Authorization": "Bearer test-token", "Protocol-Version": "1",
        "Device-Id": "device-1", "Client-Id": client_id,
    })
    await session.receive_text(json.dumps({
        "type": "hello", "version": 1, "transport": "websocket",
        "audio_params": {"format": "opus", "sample_rate": 16000,
                         "channels": 1, "frame_duration": 60},
    }))
    return session, transport

async def main():
    import xiaozhi_companion
    xiaozhi_companion.register(Context())
    from xiaozhi_companion.adapter import XiaozhiCompanionAdapter, VoiceResponseContext

    adapter = XiaozhiCompanionAdapter(
        SimpleNamespace(extra={"token": "test-token", "transport": "loopback"}),
        session_id_factory=iter(("one", "two")).__next__,
        tts_audio_loader=lambda _path, **_kwargs: (b"\x18\x01", b"\x18\x02"),
    )
    session, transport = await negotiated(adapter, "client-1")
    token = adapter._voice_response_context.set(VoiceResponseContext(session))
    try:
        assert adapter.prepare_tts_text("Hello **there**")
        result = await adapter.play_tts("device-1", "/not/read")
        assert result.success
        result = await adapter.send(
            "device-1", "Hello **there**", metadata={"notify": True}
        )
        assert result.success
    finally:
        adapter._voice_response_context.reset(token)
    assert [m.get("state") for m in transport.json_messages[1:]] == [
        "start", "sentence_start", "stop"
    ]
    assert transport.binary_messages == [b"\x18\x01", b"\x18\x02"]
    assert not any(m.get("type") == "stt" for m in transport.json_messages[1:])
    result = await adapter.send("device-1", "recognized request")
    assert result.success and transport.json_messages[-1]["type"] == "stt"

    old = session
    replacement, replacement_transport = await negotiated(adapter, "client-2")
    token = adapter._voice_response_context.set(VoiceResponseContext(old, "stale"))
    try:
        result = await adapter.play_tts("device-1", "/not/read")
        assert not result.success
    finally:
        adapter._voice_response_context.reset(token)
    assert replacement.is_active
    assert replacement_transport.binary_messages == []

asyncio.run(main())
'''
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            (str(PLUGIN_PARENT), str(HERMES_ROOT), env.get("PYTHONPATH", ""))
        )
        result = subprocess.run(
            [str(HERMES_PYTHON), "-c", script],
            cwd=HERMES_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
