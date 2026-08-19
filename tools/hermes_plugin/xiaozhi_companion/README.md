# Xiaozhi Companion Hermes plugin

This repository-owned plugin uses the public Hermes Platform Adapter API. It
does not modify Hermes core.

Ticket 06 provides a transport-independent Xiaozhi WebSocket protocol v1 core,
an explicit in-memory loopback transport, and a deterministic desktop Protocol
Harness. Ticket 07 adds a production WSS listener through the public
`websockets` 15 API. It does not modify Hermes core.

Production configuration uses `extra.transport: wss`. Set `bind_host`, `port`,
and `path`, and set absolute external paths in `cert_file`, `key_file`, and
`device_tokens_file`. The token file is a JSON object keyed by the exact
`Device-Id`. Values may contain a complete bearer value or a raw token. Keep
the certificate private key and tokens outside this repository.

`extra.transport: loopback` remains available for the Protocol Harness. It
does not start a listener unless `listener_enabled: true` is also set. An
enabled plaintext listener rejects every non-loopback bind address.

The companion firmware reads its runtime connection policy from the
`companion` NVS namespace:

- `mode`: `physical` or `loopback`
- `mdns_host`: the same-LAN host name, such as `hermes.local`
- `port` and `path`: the WSS listener endpoint
- `fallback_url`: the runtime fallback URL
- `ca_pem`: the PEM trust anchor used to verify the Hermes certificate

Physical mode fails closed when `ca_pem` is missing or an endpoint is not
`wss://`. Plaintext device transport accepts only an explicit loopback URL.
The existing `websocket/token` NVS value remains the device bearer credential.
No credential or certificate is compiled into the firmware.

The following work is intentionally outside this slice:

- Opus decoding, audio-file creation, and production STT
- Hermes installation or service configuration
- Device flashing and physical audio validation
