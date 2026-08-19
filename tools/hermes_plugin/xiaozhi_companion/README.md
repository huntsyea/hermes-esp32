# Xiaozhi Companion Hermes plugin

This repository-owned plugin uses the public Hermes Platform Adapter API. It
does not modify Hermes core.

Ticket 06 provides a transport-independent Xiaozhi WebSocket protocol v1 core,
an explicit in-memory loopback transport, and a deterministic desktop Protocol
Harness. It proves the hello, push-to-talk, Opus packet, Hermes message,
transcript, emotion, and TTS lifecycle contracts without a network listener or
physical device.

The adapter reports connected in this slice only when its platform config has
`extra.transport: loopback`. This explicit opt-in prevents the harness-only
adapter from claiming that a production network listener is active.

The following work is intentionally outside this slice:

- LAN WebSocket binding, TLS, mDNS, and reconnect policy
- Opus decoding, audio-file creation, and production STT
- Hermes installation or service configuration
- Device flashing and physical audio validation
