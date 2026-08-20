"""Secure WebSocket listener for the Xiaozhi companion platform adapter."""

from __future__ import annotations

import ipaddress
import json
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .protocol import AuthenticationError, ProtocolError, XiaozhiProtocolServer


DEFAULT_PORT = 8765
DEFAULT_PATH = "/xiaozhi/v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ListenerPolicyError(ValueError):
    pass


def _absolute_existing_file(value: Any, field: str) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute():
        raise ListenerPolicyError(f"{field}_must_be_absolute")
    resolved = path.resolve()
    if not resolved.is_file():
        raise ListenerPolicyError(f"{field}_not_found")
    if resolved.is_relative_to(REPOSITORY_ROOT):
        raise ListenerPolicyError(f"{field}_must_be_external")
    return resolved


def _is_loopback_host(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class ListenerConfig:
    transport: str
    bind_host: str
    port: int
    path: str
    cert_file: Path | None = None
    key_file: Path | None = None

    @property
    def secure(self) -> bool:
        return self.transport == "wss"

    @classmethod
    def from_extra(cls, extra: Mapping[str, Any]) -> "ListenerConfig":
        transport = str(extra.get("transport", "")).strip().casefold()
        if transport not in {"wss", "loopback"}:
            raise ListenerPolicyError("transport_must_be_wss_or_loopback")

        default_host = "0.0.0.0" if transport == "wss" else "127.0.0.1"
        bind_host = str(extra.get("bind_host", default_host)).strip()
        if transport == "loopback" and not _is_loopback_host(bind_host):
            raise ListenerPolicyError("plaintext_requires_loopback_bind")

        try:
            port = int(extra.get("port", DEFAULT_PORT))
        except (TypeError, ValueError) as error:
            raise ListenerPolicyError("invalid_port") from error
        if not 0 <= port <= 65535:
            raise ListenerPolicyError("invalid_port")

        path = str(extra.get("path", DEFAULT_PATH)).strip()
        if not path.startswith("/") or "?" in path or "#" in path:
            raise ListenerPolicyError("invalid_path")

        cert_file = None
        key_file = None
        if transport == "wss":
            cert_file = _absolute_existing_file(extra.get("cert_file"), "cert_file")
            key_file = _absolute_existing_file(extra.get("key_file"), "key_file")
        elif extra.get("cert_file") or extra.get("key_file"):
            raise ListenerPolicyError("plaintext_must_not_load_tls_keys")

        return cls(transport, bind_host, port, path, cert_file, key_file)


class FileTokenResolver:
    """Load per-device bearer tokens once from an external JSON file."""

    def __init__(self, path: Path | str):
        resolved = _absolute_existing_file(path, "device_tokens_file")
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ListenerPolicyError("invalid_device_tokens_file") from error
        if not isinstance(payload, dict) or not payload:
            raise ListenerPolicyError("device_tokens_required")

        tokens: dict[str, str] = {}
        for device_id, token_value in payload.items():
            identity = str(device_id).strip()
            token = str(token_value).strip()
            if not identity or not token:
                raise ListenerPolicyError("invalid_device_token_entry")
            tokens[identity] = token if " " in token else f"Bearer {token}"
        self._tokens = tokens

    def __call__(self, device_id: str) -> str | None:
        return self._tokens.get(device_id)


def build_server_ssl_context(
    config: ListenerConfig,
    *,
    context_factory: Callable[[], Any] | None = None,
):
    if not config.secure:
        return None
    if config.cert_file is None or config.key_file is None:
        raise ListenerPolicyError("tls_paths_required")
    factory = context_factory or (lambda: ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER))
    context = factory()
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(str(config.cert_file), str(config.key_file))
    return context


class WebsocketSessionTransport:
    def __init__(self, connection):
        self._connection = connection

    async def send_json(self, message: Mapping[str, Any]) -> None:
        await self._connection.send(json.dumps(dict(message), separators=(",", ":")))

    async def send_binary(self, payload: bytes) -> None:
        await self._connection.send(payload)

    async def close(self, code: int, reason: str) -> None:
        await self._connection.close(code=code, reason=reason)


class WebsocketListener:
    """Own one websockets 15 listener without coupling it to Hermes internals."""

    def __init__(
        self,
        config: ListenerConfig,
        protocol: XiaozhiProtocolServer,
        *,
        serve_factory=None,
        ssl_context_factory: Callable[[], Any] | None = None,
    ):
        self._config = config
        self._protocol = protocol
        self._serve_factory = serve_factory
        self._ssl_context_factory = ssl_context_factory
        self._server = None

    @property
    def bound_port(self) -> int | None:
        if self._server is None or not getattr(self._server, "sockets", None):
            return None
        return int(self._server.sockets[0].getsockname()[1])

    async def start(self) -> None:
        if self._server is not None:
            return
        serve_factory = self._serve_factory
        if serve_factory is None:
            from websockets.asyncio.server import serve

            serve_factory = serve
        ssl_context = build_server_ssl_context(
            self._config, context_factory=self._ssl_context_factory
        )
        self._server = await serve_factory(
            self._handle_connection,
            self._config.bind_host,
            self._config.port,
            ssl=ssl_context,
            compression=None,
            max_size=64 * 1024,
            ping_interval=20,
            ping_timeout=20,
        )

    async def stop(self) -> None:
        if self._server is None:
            return
        server = self._server
        self._server = None
        server.close()
        await server.wait_closed()

    async def _handle_connection(self, connection) -> None:
        request = connection.request
        if request.path != self._config.path:
            await connection.close(code=1008, reason="invalid_path")
            return
        transport = WebsocketSessionTransport(connection)
        try:
            session = await self._protocol.open_session(transport, request.headers)
        except (AuthenticationError, ProtocolError) as error:
            await connection.close(code=1008, reason=error.code)
            return

        try:
            async for message in connection:
                if isinstance(message, bytes):
                    await session.receive_binary(message)
                else:
                    await session.receive_text(message)
        except ProtocolError as error:
            await connection.close(code=1008, reason=error.code)
        finally:
            await session.disconnect()
