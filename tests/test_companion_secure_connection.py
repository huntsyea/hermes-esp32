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


ROOT = Path(__file__).resolve().parents[2]
PLUGIN_PARENT = ROOT / "tools" / "hermes_plugin"
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


class CompanionDevicePolicyTests(unittest.TestCase):
    def test_connection_policy_backoff_and_ui_wiring_contract(self):
        test_program = textwrap.dedent(
            r"""
            #include "protocols/companion_connection.h"
            #include "protocols/reconnect_controller.h"

            #include <cassert>
            #include <cstdint>
            #include <string>

            int main() {
                CompanionConnectionConfig config;
                config.mode = CompanionConnectionMode::kPhysical;
                config.mdns_host = "hermes.local";
                config.port = 8765;
                config.path = "/xiaozhi/v1";
                config.fallback_url = "wss://192.168.1.20:8765/xiaozhi/v1";
                config.ca_certificate_pem = "-----BEGIN CERTIFICATE-----\nfixture\n";

                auto discovered = ResolveCompanionEndpoint(
                    config, [](const std::string& host) { return host == "hermes.local"; });
                assert(discovered.ok());
                assert(discovered.url == "wss://hermes.local:8765/xiaozhi/v1");
                assert(discovered.source == CompanionEndpointSource::kMdns);

                auto fallback = ResolveCompanionEndpoint(
                    config, [](const std::string&) { return false; });
                assert(fallback.ok());
                assert(fallback.url == config.fallback_url);
                assert(fallback.source == CompanionEndpointSource::kFallback);

                config.ca_certificate_pem.clear();
                assert(!ResolveCompanionEndpoint(config, [](const std::string&) { return true; }).ok());
                config.ca_certificate_pem = "ca";
                config.mdns_host = "invalid/host.local";
                assert(!ResolveCompanionEndpoint(config, [](const std::string&) { return true; }).ok());
                config.mdns_host = "hermes.local";
                config.fallback_url = "ws://192.168.1.20:8765/xiaozhi/v1";
                assert(!ResolveCompanionEndpoint(config, [](const std::string&) { return false; }).ok());

                config.mode = CompanionConnectionMode::kLoopbackDevelopment;
                config.fallback_url = "ws://127.0.0.1:8765/xiaozhi/v1";
                config.mdns_host.clear();
                assert(ResolveCompanionEndpoint(config, [](const std::string&) { return false; }).ok());
                config.fallback_url = "ws://192.168.1.20:8765/xiaozhi/v1";
                assert(!ResolveCompanionEndpoint(config, [](const std::string&) { return false; }).ok());

                uint64_t now = 1000;
                uint32_t random = 0;
                ReconnectController reconnect(
                    [&now]() { return now; },
                    [&random]() { return random; },
                    ReconnectPolicy{1000, 8000, 20});
                assert(reconnect.Ready());
                auto first = reconnect.RecordFailure();
                assert(first >= 800 && first <= 1200);
                assert(!reconnect.Ready());
                now += first;
                assert(reconnect.Ready());
                random = UINT32_MAX;
                auto second = reconnect.RecordFailure();
                assert(second >= 1600 && second <= 2400);
                auto third = reconnect.RecordFailure();
                auto fourth = reconnect.RecordFailure();
                auto fifth = reconnect.RecordFailure();
                assert(third >= second);
                assert(fourth >= third);
                assert(fifth <= 9600);
                reconnect.RecordSuccess();
                assert(reconnect.attempt() == 0);
                assert(reconnect.Ready());
                return 0;
            }
            """
        )

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source = directory_path / "connection_test.cc"
            executable = directory_path / "connection_test"
            source.write_text(test_program, encoding="utf-8")
            completed = subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(ROOT / "main"),
                    str(source),
                    str(ROOT / "main" / "protocols" / "companion_connection.cc"),
                    str(ROOT / "main" / "protocols" / "reconnect_controller.cc"),
                    "-o",
                    str(executable),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"{completed.stdout}\n{completed.stderr}",
            )
            subprocess.run([str(executable)], timeout=10, check=True)

        display_header = (ROOT / "main" / "display" / "display.h").read_text(
            encoding="utf-8"
        )
        application = (ROOT / "main" / "application.cc").read_text(encoding="utf-8")
        board = (
            ROOT
            / "main"
            / "boards"
            / "waveshare"
            / "esp32-s3-touch-lcd-1.54"
            / "esp32-s3-touch-lcd-1.54.cc"
        ).read_text(encoding="utf-8")
        websocket = (ROOT / "main" / "protocols" / "websocket_protocol.cc").read_text(
            encoding="utf-8"
        )

        self.assertIn("SetConversationEvent", display_header)
        self.assertIn("ConversationEvent::kConnected", application)
        self.assertIn("ConversationEvent::kDisconnected", application)
        self.assertIn("screen_manager_dispatch_event", board)
        self.assertIn("if (!Lock(1))", board)
        self.assertIn("reconnect_.RecordFailure", websocket)
        self.assertIn("reconnect_.RecordSuccess", websocket)
        self.assertIn("secure_websocket", websocket)
        self.assertIn("!intentional_close_.load() && on_disconnected_", websocket)

    def test_physical_websocket_connects_off_main_task_with_bounded_tx(self):
        websocket = (ROOT / "main" / "protocols" / "websocket_protocol.cc").read_text(
            encoding="utf-8"
        )
        secure = (
            ROOT / "main" / "protocols" / "secure_websocket_client.cc"
        ).read_text(encoding="utf-8")

        self.assertIn("StartSecureConnection", websocket)
        self.assertIn("StartSecureShutdown", websocket)
        self.assertIn("DeferSecureShutdown(std::move(secure))", websocket)
        self.assertIn("xTaskCreate", websocket)
        self.assertIn("FinishSecureConnection", websocket)
        self.assertIn("context.reset()", websocket)
        self.assertIn("kSendTimeout", secure)
        self.assertNotIn("pdMS_TO_TICKS(10000))", secure.split("bool SecureWebSocketClient::Send", 1)[1])
        cancel = secure.split("void SecureWebSocketClient::Cancel()", 1)[1].split(
            "bool SecureWebSocketClient::IsConnected", 1
        )[0]
        self.assertNotIn("Shutdown()", cancel)
        send = secure.split("bool SecureWebSocketClient::Send(const void*", 1)[1].split(
            "void SecureWebSocketClient::HandleEvent", 1
        )[0]
        self.assertIn("cancelled_.load()", send)
        self.assertIn("secure_connection_lifecycle_mutex_", websocket)

    def test_secure_close_linearizes_hello_send_and_connected_publish(self):
        source = textwrap.dedent(
            r"""
            #include <atomic>
            #include <cassert>
            #include <mutex>
            #include <thread>

            class ConnectionLifecycle {
             public:
              bool SendHello(unsigned expected_generation) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (generation_ != expected_generation || cancelled_) {
                  return false;
                }
                assert(!close_returned_.load());
                hello_sent_ = true;
                return true;
              }

              bool PublishConnected(unsigned expected_generation) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (generation_ != expected_generation || cancelled_) {
                  return false;
                }
                assert(!close_returned_.load());
                connected_ = true;
                return true;
              }

              void Close() {
                std::lock_guard<std::mutex> lock(mutex_);
                ++generation_;
                cancelled_ = true;
                connected_ = false;
                close_returned_.store(true);
              }

              bool connected() const { return connected_; }

             private:
              std::mutex mutex_;
              unsigned generation_ = 1;
              bool cancelled_ = false;
              bool hello_sent_ = false;
              bool connected_ = false;
              std::atomic<bool> close_returned_{false};
            };

            int main() {
              for (int iteration = 0; iteration < 1000; ++iteration) {
                ConnectionLifecycle lifecycle;
                std::thread worker([&] {
                  lifecycle.SendHello(1);
                  std::this_thread::yield();
                  lifecycle.PublishConnected(1);
                });
                std::thread closer([&] { lifecycle.Close(); });
                worker.join();
                closer.join();
                assert(!lifecycle.connected());
              }
              return 0;
            }
            """
        )
        with tempfile.TemporaryDirectory() as directory:
            cpp = Path(directory) / "secure_lifecycle_race.cc"
            executable = Path(directory) / "secure_lifecycle_race"
            cpp.write_text(source, encoding="utf-8")
            subprocess.run(
                ["c++", "-std=c++17", "-pthread", str(cpp), "-o", str(executable)],
                check=True,
            )
            subprocess.run([str(executable)], timeout=10, check=True)

    def test_network_error_mutation_is_scheduled_on_application_task(self):
        application = (ROOT / "main" / "application.cc").read_text(encoding="utf-8")
        handler = application.split("protocol_->OnNetworkError", 1)[1].split(
            "protocol_->OnIncomingAudio", 1
        )[0]
        self.assertIn("Schedule([this, message]", handler)

    def test_legacy_hello_stays_permissive_and_strict_mode_requires_identity(self):
        websocket = (ROOT / "main" / "protocols" / "websocket_protocol.cc").read_text(
            encoding="utf-8"
        )

        hello = websocket.split("void WebsocketProtocol::ParseServerHello", 1)[1]
        self.assertIn("strict_session_validation_", hello)
        self.assertIn("legacy-", hello)
        self.assertIn("synthesized_legacy_session_", websocket)
        self.assertIn('\\"session_id\\":\\"\\"', websocket)
        self.assertIn("if (cJSON_IsObject(audio_params))", hello)

    def test_generation_changes_after_reconnect_policy_accepts_attempt(self):
        websocket = (ROOT / "main" / "protocols" / "websocket_protocol.cc").read_text(
            encoding="utf-8"
        )

        open_channel = websocket.split("bool WebsocketProtocol::OpenAudioChannel()", 1)[1].split(
            "bool WebsocketProtocol::IsTransportConnected", 1
        )[0]
        self.assertLess(open_channel.index("reconnect_.Ready()"), open_channel.index("transport_generation_"))
        active_attempt = open_channel.split("IsSecureConnectionTaskActive()", 1)[1].split("}", 1)[0]
        self.assertIn("return false", active_attempt)


@unittest.skipUnless(
    HERMES_PYTHON.is_file()
    and (HERMES_ROOT / "gateway" / "platforms" / "base.py").is_file(),
    "live Hermes checkout and venv are not available",
)
class LiveHermesSecurePluginTests(unittest.TestCase):
    def test_secure_plugin_config_loads_with_live_hermes_and_websockets(self):
        script = r'''
import importlib
from types import SimpleNamespace

from gateway.platform_registry import PlatformEntry


class Context:
    def __init__(self):
        self.entry = None

    def register_platform(self, **kwargs):
        self.entry = PlatformEntry(source="plugin", **kwargs)


module = importlib.import_module("xiaozhi_companion")
context = Context()
module.register(context)
assert context.entry.check_fn()
assert not context.entry.validate_config(SimpleNamespace(extra={
    "transport": "wss",
    "bind_host": "0.0.0.0",
    "cert_file": "/external/server.pem",
    "key_file": "/external/server.key",
    "device_tokens_file": "/external/devices.json",
}))
print("secure-plugin-doctor-ok")
'''
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(PLUGIN_PARENT), str(HERMES_ROOT))
        )
        completed = subprocess.run(
            [str(HERMES_PYTHON), "-c", script],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"{completed.stdout}\n{completed.stderr}",
        )
        self.assertIn("secure-plugin-doctor-ok", completed.stdout)

    def test_validate_config_parses_tls_material_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def generate_pair(name):
                cert = root / f"{name}.pem"
                key = root / f"{name}.key"
                completed = subprocess.run(
                    [
                        "openssl",
                        "req",
                        "-x509",
                        "-newkey",
                        "rsa:2048",
                        "-nodes",
                        "-days",
                        "1",
                        "-subj",
                        "/CN=localhost",
                        "-keyout",
                        str(key),
                        "-out",
                        str(cert),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, msg=completed.stderr)
                return cert, key

            cert, key = generate_pair("server")
            _other_cert, other_key = generate_pair("other")
            malformed_cert = root / "malformed.pem"
            malformed_cert.write_text("not a certificate", encoding="utf-8")
            tokens = root / "device-tokens.json"
            tokens.write_text(json.dumps({"device-1": "secret"}), encoding="utf-8")

            script = r'''
import importlib
import sys
from types import SimpleNamespace

from gateway.platform_registry import PlatformEntry


class Context:
    def __init__(self):
        self.entry = None

    def register_platform(self, **kwargs):
        self.entry = PlatformEntry(source="plugin", **kwargs)


module = importlib.import_module("xiaozhi_companion")
context = Context()
module.register(context)


def config(cert, key):
    return SimpleNamespace(extra={
        "transport": "wss",
        "bind_host": "127.0.0.1",
        "port": 0,
        "cert_file": cert,
        "key_file": key,
        "device_tokens_file": sys.argv[4],
    })


assert context.entry.validate_config(config(sys.argv[1], sys.argv[2]))
assert not context.entry.validate_config(config(sys.argv[3], sys.argv[2]))
assert not context.entry.validate_config(config(sys.argv[1], sys.argv[5]))
print("tls-preflight-ok")
'''
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                (str(PLUGIN_PARENT), str(HERMES_ROOT))
            )
            completed = subprocess.run(
                [
                    str(HERMES_PYTHON),
                    "-c",
                    script,
                    str(cert),
                    str(key),
                    str(malformed_cert),
                    str(tokens),
                    str(other_key),
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"{completed.stdout}\n{completed.stderr}",
            )
            self.assertIn("tls-preflight-ok", completed.stdout)

    def test_ephemeral_wss_listener_negotiates_and_rejects_wrong_device_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cert = root / "server.pem"
            key = root / "server.key"
            openssl_config = root / "openssl.cnf"
            openssl_config.write_text(
                textwrap.dedent(
                    """
                    [req]
                    distinguished_name = subject
                    x509_extensions = extensions
                    prompt = no
                    [subject]
                    CN = localhost
                    [extensions]
                    subjectAltName = DNS:localhost
                    keyUsage = digitalSignature,keyEncipherment
                    extendedKeyUsage = serverAuth
                    """
                ),
                encoding="utf-8",
            )
            generated = subprocess.run(
                [
                    "openssl",
                    "req",
                    "-x509",
                    "-newkey",
                    "rsa:2048",
                    "-nodes",
                    "-days",
                    "1",
                    "-config",
                    str(openssl_config),
                    "-keyout",
                    str(key),
                    "-out",
                    str(cert),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(generated.returncode, 0, msg=generated.stderr)

            script = r'''
import asyncio
import json
import ssl
import sys

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from xiaozhi_companion.network import ListenerConfig, WebsocketListener
from xiaozhi_companion.protocol import XiaozhiProtocolServer


async def main(cert, key):
    session_ids = iter(("ephemeral-session-1", "ephemeral-session-2"))
    listener = WebsocketListener(
        ListenerConfig(
            transport="wss",
            bind_host="127.0.0.1",
            port=0,
            path="/xiaozhi/v1",
            cert_file=__import__("pathlib").Path(cert),
            key_file=__import__("pathlib").Path(key),
        ),
        XiaozhiProtocolServer(
            token_resolver=lambda device_id: (
                "Bearer ephemeral-token" if device_id == "device-a" else None
            ),
            on_voice_message=lambda _message: None,
            session_id_factory=lambda: next(session_ids),
        ),
    )
    await listener.start()
    context = ssl.create_default_context(cafile=cert)
    uri = f"wss://localhost:{listener.bound_port}/xiaozhi/v1"
    headers = {
        "Authorization": "Bearer ephemeral-token",
        "Protocol-Version": "1",
        "Device-Id": "device-a",
        "Client-Id": "client-a",
    }
    try:
        async with connect(uri, ssl=context, additional_headers=headers) as first_socket:
            await first_socket.send(json.dumps({
                "type": "hello",
                "version": 1,
                "transport": "websocket",
                "audio_params": {
                    "format": "opus",
                    "sample_rate": 16000,
                    "channels": 1,
                    "frame_duration": 60,
                },
            }))
            hello = json.loads(await first_socket.recv())
            assert hello["session_id"] == "ephemeral-session-1"

            async with connect(uri, ssl=context, additional_headers=headers) as replacement_socket:
                await replacement_socket.send(json.dumps({
                    "type": "hello",
                    "version": 1,
                    "transport": "websocket",
                    "audio_params": {
                        "format": "opus",
                        "sample_rate": 16000,
                        "channels": 1,
                        "frame_duration": 60,
                    },
                }))
                replacement_hello = json.loads(await replacement_socket.recv())
                assert replacement_hello["session_id"] == "ephemeral-session-2"
                try:
                    await first_socket.recv()
                except ConnectionClosed as error:
                    assert error.code == 4001
                else:
                    raise AssertionError("stale device session was not replaced")

        bad_headers = dict(headers)
        bad_headers["Authorization"] = "Bearer wrong-token"
        async with connect(uri, ssl=context, additional_headers=bad_headers) as socket:
            try:
                await socket.recv()
            except ConnectionClosed as error:
                assert error.code == 1008
            else:
                raise AssertionError("wrong token was not rejected")
    finally:
        await listener.stop()


asyncio.run(main(sys.argv[1], sys.argv[2]))
print("ephemeral-wss-ok")
'''
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                (str(PLUGIN_PARENT), str(HERMES_ROOT))
            )
            completed = subprocess.run(
                [str(HERMES_PYTHON), "-c", script, str(cert), str(key)],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"{completed.stdout}\n{completed.stderr}",
            )
            self.assertIn("ephemeral-wss-ok", completed.stdout)


if __name__ == "__main__":
    unittest.main()
