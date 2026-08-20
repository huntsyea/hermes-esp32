"""Dependency-free Ogg container writer for fixed-duration Opus packets."""

from __future__ import annotations

import struct
from collections.abc import Iterable


MAX_PACKET_BYTES = 16 * 1024
MAX_PACKET_COUNT = 5000
MAX_RECORDING_BYTES = 4 * 1024 * 1024
OPUS_GRANULE_RATE = 48_000
DEFAULT_FRAME_DURATION_MS = 60


class OggOpusError(ValueError):
    """Raised when packets cannot be represented by this bounded writer."""


def ogg_crc32(payload: bytes) -> int:
    checksum = 0
    for byte in payload:
        checksum ^= byte << 24
        for _ in range(8):
            if checksum & 0x80000000:
                checksum = ((checksum << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
            else:
                checksum = (checksum << 1) & 0xFFFFFFFF
    return checksum


def _lacing(packet_size: int) -> bytes:
    values = bytearray()
    remaining = packet_size
    while remaining >= 255:
        values.append(255)
        remaining -= 255
    values.append(remaining)
    if len(values) > 255:
        raise OggOpusError("audio_packet_too_large")
    return bytes(values)


def _page(
    packet: bytes,
    *,
    flags: int,
    granule_position: int,
    serial: int,
    sequence: int,
) -> bytes:
    lacing = _lacing(len(packet))
    header = bytearray(
        b"OggS"
        + bytes((0, flags))
        + struct.pack("<QII", granule_position, serial, sequence)
        + b"\0\0\0\0"
        + bytes((len(lacing),))
        + lacing
    )
    page = header + packet
    struct.pack_into("<I", page, 22, ogg_crc32(page))
    return bytes(page)


def packets_to_ogg_opus(
    packets: Iterable[bytes],
    *,
    serial: int,
    input_sample_rate: int = 16_000,
    channels: int = 1,
    frame_duration_ms: int = DEFAULT_FRAME_DURATION_MS,
) -> bytes:
    """Wrap fixed-duration Opus packets in a deterministic Ogg stream."""
    frames = tuple(packets)
    if not frames:
        raise OggOpusError("empty_recording")
    if len(frames) > MAX_PACKET_COUNT:
        raise OggOpusError("too_many_audio_packets")
    if not 0 <= serial <= 0xFFFFFFFF:
        raise OggOpusError("invalid_stream_serial")
    if input_sample_rate != 16_000 or channels != 1 or frame_duration_ms != 60:
        raise OggOpusError("unsupported_audio_parameters")

    recording_bytes = 0
    for frame in frames:
        if not isinstance(frame, bytes) or not frame:
            raise OggOpusError("invalid_audio_packet")
        if len(frame) > MAX_PACKET_BYTES:
            raise OggOpusError("audio_packet_too_large")
        recording_bytes += len(frame)
        if recording_bytes > MAX_RECORDING_BYTES:
            raise OggOpusError("recording_too_large")

    # The device supplies already framed live audio, not samples from an encoder
    # whose look-ahead is known here. A zero pre-skip preserves the full capture.
    pre_skip = 0
    opus_head = b"OpusHead" + struct.pack(
        "<BBHIhB", 1, channels, pre_skip, input_sample_rate, 0, 0
    )
    vendor = b"xiaozhi-companion"
    opus_tags = b"OpusTags" + struct.pack("<I", len(vendor)) + vendor + struct.pack("<I", 0)

    pages = [
        _page(opus_head, flags=0x02, granule_position=0, serial=serial, sequence=0),
        _page(opus_tags, flags=0, granule_position=0, serial=serial, sequence=1),
    ]
    granule_increment = (OPUS_GRANULE_RATE * frame_duration_ms) // 1000
    for index, frame in enumerate(frames):
        pages.append(
            _page(
                frame,
                flags=0x04 if index == len(frames) - 1 else 0,
                granule_position=(index + 1) * granule_increment,
                serial=serial,
                sequence=index + 2,
            )
        )
    return b"".join(pages)
