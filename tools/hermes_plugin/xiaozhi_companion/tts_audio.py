"""Bounded conversion and demuxing for Companion downlink audio."""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path
from typing import Callable

from .ogg_opus import _crc


MAX_CONTAINER_BYTES = 8 * 1024 * 1024
MAX_PAGES = 5002
MAX_PACKETS = 5000
MAX_PACKET_BYTES = 16 * 1024


class TtsAudioError(ValueError):
    """Raised when TTS audio cannot be sent through the negotiated stream."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _packet_duration_ms(packet: bytes) -> int:
    if not packet:
        raise TtsAudioError("empty_opus_packet")
    config = packet[0] >> 3
    if config < 12:
        frame_ms = (10, 20, 40, 60)[config & 3]
    elif config < 16:
        frame_ms = (10, 20)[config & 1]
    else:
        frame_ms = (2.5, 5, 10, 20)[config & 3]
    count_code = packet[0] & 3
    if count_code == 0:
        frame_count = 1
    elif count_code in (1, 2):
        frame_count = 2
    else:
        if len(packet) < 2:
            raise TtsAudioError("malformed_opus_packet")
        frame_count = packet[1] & 0x3F
        if frame_count == 0:
            raise TtsAudioError("malformed_opus_packet")
    return int(frame_ms * frame_count)


def demux_ogg_opus(
    container: bytes,
    *,
    max_container_bytes: int = MAX_CONTAINER_BYTES,
    max_pages: int = MAX_PAGES,
    max_packets: int = MAX_PACKETS,
    verify_checksum: bool = True,
) -> tuple[bytes, ...]:
    """Validate one Ogg Opus stream and return only its 60 ms audio packets."""
    if not isinstance(container, bytes) or not container:
        raise TtsAudioError("empty_audio")
    if len(container) > max_container_bytes:
        raise TtsAudioError("audio_container_too_large")

    offset = 0
    expected_serial = None
    expected_sequence = 0
    packet = bytearray()
    packets: list[bytes] = []
    page_count = 0
    eos_seen = False
    while offset < len(container):
        if page_count >= max_pages or offset + 27 > len(container):
            raise TtsAudioError("malformed_ogg")
        if container[offset : offset + 4] != b"OggS" or container[offset + 4] != 0:
            raise TtsAudioError("malformed_ogg")
        flags = container[offset + 5]
        serial, sequence = struct.unpack_from("<II", container, offset + 14)
        segment_count = container[offset + 26]
        header_end = offset + 27 + segment_count
        if header_end > len(container):
            raise TtsAudioError("malformed_ogg")
        lacing = container[offset + 27 : header_end]
        page_end = header_end + sum(lacing)
        if page_end > len(container):
            raise TtsAudioError("malformed_ogg")
        if expected_serial is None:
            if not flags & 0x02:
                raise TtsAudioError("ogg_bos_required")
            expected_serial = serial
        if serial != expected_serial or sequence != expected_sequence or eos_seen:
            raise TtsAudioError("invalid_ogg_sequence")
        if verify_checksum:
            page = bytearray(container[offset:page_end])
            expected_crc = struct.unpack_from("<I", page, 22)[0]
            page[22:26] = b"\0\0\0\0"
            if _crc(page) != expected_crc:
                raise TtsAudioError("ogg_checksum_mismatch")
        body_offset = header_end
        for lace in lacing:
            packet.extend(container[body_offset : body_offset + lace])
            body_offset += lace
            if len(packet) > MAX_PACKET_BYTES:
                raise TtsAudioError("audio_packet_too_large")
            if lace < 255:
                packets.append(bytes(packet))
                packet.clear()
                if len(packets) > max_packets + 2:
                    raise TtsAudioError("too_many_audio_packets")
        eos_seen = bool(flags & 0x04)
        page_count += 1
        expected_sequence += 1
        offset = page_end

    if packet or not eos_seen or len(packets) < 3:
        raise TtsAudioError("incomplete_ogg_stream")
    opus_head, opus_tags, *audio_packets = packets
    if len(opus_head) < 19 or not opus_head.startswith(b"OpusHead"):
        raise TtsAudioError("opus_head_required")
    version, channels = opus_head[8], opus_head[9]
    sample_rate = struct.unpack_from("<I", opus_head, 12)[0]
    if version != 1 or channels != 1 or sample_rate != 16000:
        raise TtsAudioError("unsupported_audio_parameters")
    if not opus_tags.startswith(b"OpusTags"):
        raise TtsAudioError("opus_tags_required")
    if not audio_packets:
        raise TtsAudioError("empty_audio")
    if len(audio_packets) > max_packets:
        raise TtsAudioError("too_many_audio_packets")
    for audio_packet in audio_packets:
        if _packet_duration_ms(audio_packet) != 60:
            raise TtsAudioError("unsupported_frame_duration")
    return tuple(audio_packets)


def load_tts_opus_packets(
    audio_path: str | Path,
    *,
    ffmpeg_executable: str = "ffmpeg",
    runner: Callable = subprocess.run,
    max_container_bytes: int = MAX_CONTAINER_BYTES,
) -> tuple[bytes, ...]:
    """Load Ogg Opus or transcode another whole-file format without a shell."""
    path = Path(audio_path)
    try:
        source_size = path.stat().st_size
    except OSError as exc:
        raise TtsAudioError("audio_source_unavailable") from exc
    if source_size <= 0 or source_size > max_container_bytes:
        raise TtsAudioError("audio_source_size_invalid")
    try:
        source = path.read_bytes()
    except OSError as exc:
        raise TtsAudioError("audio_source_unavailable") from exc
    if source.startswith(b"OggS"):
        try:
            return demux_ogg_opus(source, max_container_bytes=max_container_bytes)
        except TtsAudioError as exc:
            if exc.code not in (
                "unsupported_audio_parameters",
                "unsupported_frame_duration",
            ):
                raise

    argv = [
        ffmpeg_executable,
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-vn",
        "-c:a",
        "libopus",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-frame_duration",
        "60",
        "-fs",
        str(max_container_bytes),
        "-f",
        "ogg",
        "pipe:1",
    ]
    try:
        result = runner(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise TtsAudioError("audio_transcode_failed") from exc
    output = bytes(result.stdout or b"")
    if result.returncode != 0 or not output:
        raise TtsAudioError("audio_transcode_failed")
    if len(output) > max_container_bytes:
        raise TtsAudioError("audio_container_too_large")
    return demux_ogg_opus(output, max_container_bytes=max_container_bytes)
