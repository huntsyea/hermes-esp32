# hermes-esp32

A [Hermes](https://github.com/NousResearch/hermes-agent) platform adapter for a pocket voice companion. It lets an ESP32 device with a screen and a mic talk to your agent over your LAN: hold to talk, then hear the answer.

The firmware lives in [esp32-companion](https://github.com/huntsyea/esp32-companion). This repo is the Hermes side. It uses the public Platform Adapter API, so Hermes core isn't modified.

## How a turn works

```
ESP32 companion ──wss:// JSON + Opus 16 kHz──▶ xiaozhi_companion adapter ──▶ Hermes gateway
       ▲                                              │
       └──── transcript, then reply audio ◀───────────┘
```

1. You hold the button. The device streams 60 ms Opus frames over a WebSocket.
2. When you let go, the adapter packages the recording as Ogg Opus and hands it to Hermes as a voice message. Hermes's own speech-to-text runs on it, and the transcript goes back to the screen.
3. Hermes answers. The adapter takes the TTS file, converts it with ffmpeg if needed, and streams it back as 16 kHz mono Opus packets.
4. If you interrupt, the adapter drops the partial recording and any audio still playing. Each turn has a generation number, so a cancelled turn can't send late audio or text.

The device speaks the [XiaoZhi](https://github.com/78/xiaozhi-esp32) WebSocket protocol v1. The wire contract shared with the firmware is in [PROTOCOL.md](PROTOCOL.md).

## Security

- The production listener is `wss://` only. It loads a certificate and key from absolute paths and rejects a mismatched pair before it starts.
- Each device authenticates with its own bearer token, looked up by `Device-Id` in a JSON file that you keep outside the repo.
- On the firmware side, the certificate's trust anchor and the token live in the device's flash storage (NVS). Nothing secret is compiled into the firmware.
- Plaintext WebSockets are only accepted on loopback, for the test harness.

## Setup

Clone this repo into your Hermes plugins directory:

```bash
git clone https://github.com/huntsyea/hermes-esp32 ~/.hermes/plugins/hermes-esp32
```

Then configure the `xiaozhi_companion` platform's `extra` settings:

| Key | Value |
| --- | --- |
| `transport` | `wss` |
| `bind_host`, `port`, `path` | Where to listen. Defaults are port `8765` and path `/xiaozhi/v1`. |
| `cert_file`, `key_file` | Absolute paths to the TLS certificate and private key |
| `device_tokens_file` | Absolute path to a JSON object mapping each `Device-Id` to its token |
| `ffmpeg_executable` | Optional. Only needed if `ffmpeg` isn't on the Hermes service's `PATH`. |

Restart the gateway. The first time a device connects, approve it with `hermes pairing approve`.

More detail on each step is in [xiaozhi_companion/README.md](xiaozhi_companion/README.md).

## Tests

```bash
python3 -m unittest discover -s tests
```

The protocol, transport and recovery tests run anywhere. The live speech-to-text and TTS tests need a working Hermes install.

## Limits

- Push-to-talk only. The adapter rejects XiaoZhi's `auto` and `realtime` listening modes with an error the device can recover from.
- LAN only. There's no relay, so the device and Hermes need to be on the same network.
- The protocol is pre-1.0. Any incompatible change bumps `PROTOCOL_VERSION` in both repos at the same time.

## Deploying to my own setup

[`scripts/deploy-to-mini.sh`](scripts/deploy-to-mini.sh) is how I ship changes to the Mac mini that runs my Hermes. It pushes HEAD over SSH, runs the tests with the runtime's Python, restarts the gateway, and waits for the adapter to reconnect. Set `MINI_HOST` and `HERMES_HOME` to point it at your own machine.
