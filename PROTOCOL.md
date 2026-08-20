# Xiaozhi Companion Wire Protocol

Shared contract between the ESP32 companion firmware (repo: `esp32-companion`) and this Hermes adapter plugin.

Wire identity:

- Transport: WebSocket, path `/xiaozhi/v1`
- `PROTOCOL_VERSION = 1` (defined in `xiaozhi_companion/protocol.py`)
- Audio: Opus, 16 kHz mono, 60 ms frames
- Control: JSON messages, session generations guard against stale/cancelled sessions
- `TRANSPORT = "websocket"`

Change rule: both repos must bump `PROTOCOL_VERSION` together for any incompatible wire change. Keep the constant in a single place per repo and reference this document from the firmware `config.json` `product_ui.protocol_version` guard.

This contract is pre-1.0 and intentionally narrow; extend only via new JSON message types or new audio session headers, never by mutating existing semantics in place.
