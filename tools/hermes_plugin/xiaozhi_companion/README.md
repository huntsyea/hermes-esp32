# Xiaozhi Companion Hermes plugin

This repository-owned plugin uses the public Hermes Platform Adapter API. It
does not modify Hermes core.

Ticket 06 provides a transport-independent Xiaozhi WebSocket protocol v1 core,
an explicit in-memory loopback transport, and a deterministic desktop Protocol
Harness. Ticket 07 adds a production WSS listener through the public
`websockets` 15 API. It does not modify Hermes core.

Ticket 08 packages each bounded push-to-talk capture as Ogg Opus, caches it
through the public Hermes media API, and emits one `VOICE` event with an
`audio/ogg` local path. This lets the standard Hermes automatic STT path
process the recording. The resulting transcript is sent to the active device
session as Xiaozhi `stt` JSON.

Ticket 09 returns one Hermes response to the same active session. The adapter
uses Hermes final-delivery metadata to keep agent responses separate from STT
echoes. Whole-file TTS is validated as Ogg Opus or converted with a configured
`ffmpeg_executable`, then demuxed into bounded 16 kHz mono 60 ms Opus packets.
Only those packets are sent between the Xiaozhi `tts` start, sentence, and
stop messages; Ogg headers never enter the device audio stream.

Production configuration uses `extra.transport: wss`. Set `bind_host`, `port`,
and `path`, and set absolute external paths in `cert_file`, `key_file`, and
`device_tokens_file`. The token file is a JSON object keyed by the exact
`Device-Id`. Values may contain a complete bearer value or a raw token. Keep
the certificate private key and tokens outside this repository.
Set `ffmpeg_executable` only when `ffmpeg` is not available on the Hermes
service PATH.

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

- Live STT-provider validation with a configured Hermes service
- Live TTS-provider and speaker validation with a configured Hermes service
- Hermes installation or service configuration
- Device flashing and physical audio validation
