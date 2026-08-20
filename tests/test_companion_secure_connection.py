import asyncio
import json
import os
import ssl
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_PARENT = ROOT
HERMES_ROOT = Path(
    os.environ.get("HERMES_AGENT_ROOT", Path.home() / ".hermes" / "hermes-agent")
)
HERMES_PYTHON = HERMES_ROOT / "venv" / "bin" / "python"

sys.path.insert(0, str(PLUGIN_PARENT))


class DeviceAuthenticationTests(unittest.IsolatedAsyncioTestCase):
    async def test_tokens_are_resolved_per_device_and_compared_without_leaking(self):
        from xiaozhi_companion.loopback import LoopbackTransport
        from xiaozhi_companion.protocol import AuthenticationError, XiaozhiProtocolServer

        diagnostics = []
        server = XiaozhiProtocolServer(
            token_resolver=lambda device_id: {
                "device-a": "Bearer token-a",
                "device-b": "Bearer token-b",
            }.get(device_id),
            on_voice_message=lambda _message: None,
            diagnostic_sink=diagnostics.append,
            session_id_factory=lambda: "session-secure",
        )

        headers = {
            "Authorization": "Bearer token-a",
            "Protocol-Version": "1",
            "Device-Id": "device-a",
            "Client-Id": "client-a",
        }
        session = await server.open_session(LoopbackTransport(), headers)
        self.assertEqual(session.device_id, "device-a")

        for device_id, token in (
            ("device-b", "Bearer token-a"),
            ("unknown", "Bearer token-a"),
            ("device-a", ""),
        ):
            rejected_headers = dict(headers, **{"Device-Id": device_id})
            rejected_headers["Authorization"] = token
            with self.assertRaises(AuthenticationError):
                await server.open_session(LoopbackTransport(), rejected_headers)

        rendered = repr(diagnostics)
        self.assertNotIn("token-a", rendered)
        self.assertNotIn("token-b", rendered)

    async def test_unknown_device_still_uses_constant_time_comparison(self):
        from xiaozhi_companion.loopback import LoopbackTransport
        from xiaozhi_companion.protocol import AuthenticationError, XiaozhiProtocolServer

        server = XiaozhiProtocolServer(
            token_resolver=lambda _device_id: None,
            on_voice_message=lambda _message: None,
        )
        headers = {
            "Authorization": "Bearer rejected-token",
            "Protocol-Version": "1",
            "Device-Id": "unknown-device",
            "Client-Id": "client-a",
        }

        with mock.patch(
            "xiaozhi_companion.protocol.hmac.compare_digest", return_value=False
        ) as compare_digest:
            with self.assertRaises(AuthenticationError):
                await server.open_session(LoopbackTransport(), headers)

        compare_digest.assert_called_once()


class HermesListenerPolicyTests(unittest.TestCase):
    def test_wss_requires_external_certificate_and_key_paths(self):
        from xiaozhi_companion.network import ListenerConfig, ListenerPolicyError

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cert = root / "server.pem"
            key = root / "server.key"
            cert.write_text("test certificate fixture", encoding="utf-8")
            key.write_text("test private key fixture", encoding="utf-8")

            config = ListenerConfig.from_extra(
                {
                    "transport": "wss",
                    "bind_host": "0.0.0.0",
                    "port": 8765,
                    "path": "/xiaozhi/v1",
                    "cert_file": str(cert),
                    "key_file": str(key),
                }
            )
            self.assertTrue(config.secure)
            self.assertEqual(config.path, "/xiaozhi/v1")
            self.assertEqual(config.cert_file, cert.resolve())
            self.assertEqual(config.key_file, key.resolve())

            for extra in (
                {"transport": "wss", "cert_file": str(cert)},
                {"transport": "wss", "key_file": str(key)},
                {"transport": "wss", "cert_file": "relative.pem", "key_file": str(key)},
            ):
                with self.assertRaises(ListenerPolicyError):
                    ListenerConfig.from_extra(extra)

    def test_plaintext_listener_is_explicit_and_loopback_only(self):
        from xiaozhi_companion.network import ListenerConfig, ListenerPolicyError

        config = ListenerConfig.from_extra(
            {
                "transport": "loopback",
                "bind_host": "127.0.0.1",
                "port": 8765,
                "path": "/xiaozhi/v1",
            }
        )
        self.assertFalse(config.secure)

        for host in ("0.0.0.0", "192.168.1.10", "hermes.local"):
            with self.assertRaises(ListenerPolicyError):
                ListenerConfig.from_extra(
                    {"transport": "loopback", "bind_host": host}
                )

        with self.assertRaises(ListenerPolicyError):
            ListenerConfig.from_extra({"transport": "ws", "bind_host": "127.0.0.1"})

    def test_ssl_context_loads_only_configured_external_paths(self):
        from xiaozhi_companion.network import ListenerConfig, build_server_ssl_context

        calls = []

        class FakeContext:
            minimum_version = None

            def load_cert_chain(self, certfile, keyfile):
                calls.append((certfile, keyfile))

        config = ListenerConfig(
            transport="wss",
            bind_host="0.0.0.0",
            port=8765,
            path="/xiaozhi/v1",
            cert_file=Path("/external/server.pem"),
            key_file=Path("/external/server.key"),
        )
        context = build_server_ssl_context(config, context_factory=FakeContext)

        self.assertIsInstance(context, FakeContext)
        self.assertEqual(context.minimum_version, ssl.TLSVersion.TLSv1_2)
        self.assertEqual(calls, [("/external/server.pem", "/external/server.key")])

    def test_device_token_file_is_external_and_keyed_by_device_id(self):
        from xiaozhi_companion.network import FileTokenResolver, ListenerPolicyError

        with tempfile.TemporaryDirectory() as directory:
            token_path = Path(directory) / "device-tokens.json"
            token_path.write_text(
                json.dumps({"device-a": "token-a", "device-b": "Bearer token-b"}),
                encoding="utf-8",
            )
            resolver = FileTokenResolver(token_path)
            self.assertEqual(resolver("device-a"), "Bearer token-a")
            self.assertEqual(resolver("device-b"), "Bearer token-b")
            self.assertIsNone(resolver("missing"))

            with self.assertRaises(ListenerPolicyError):
                FileTokenResolver(Path("relative.json"))

    def test_listener_uses_websockets_15_public_serve_contract_without_binding(self):
        from xiaozhi_companion.network import ListenerConfig, WebsocketListener
        from xiaozhi_companion.protocol import XiaozhiProtocolServer

        calls = []

        class FakeServer:
            def close(self):
                calls.append("close")

            async def wait_closed(self):
                calls.append("wait_closed")

        async def fake_serve(handler, host, port, **kwargs):
            calls.append((handler, host, port, kwargs))
            return FakeServer()

        async def exercise():
            listener = WebsocketListener(
                ListenerConfig(
                    transport="loopback",
                    bind_host="127.0.0.1",
                    port=0,
                    path="/xiaozhi/v1",
                ),
                XiaozhiProtocolServer(
                    expected_token="Bearer test-token",
                    on_voice_message=lambda _message: None,
                ),
                serve_factory=fake_serve,
            )
            await listener.start()
            await listener.stop()

        asyncio.run(exercise())
        handler, host, port, kwargs = calls[0]
        self.assertTrue(callable(handler))
        self.assertEqual((host, port), ("127.0.0.1", 0))
        self.assertIsNone(kwargs["ssl"])
        self.assertEqual(kwargs["compression"], None)
        self.assertEqual(calls[1:], ["close", "wait_closed"])


