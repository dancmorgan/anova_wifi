import asyncio
import json
import logging
import uuid
from asyncio import Future
from datetime import UTC, datetime
from typing import Any

from aiohttp import (
    ClientConnectionError,
    ClientConnectionResetError,
    ClientSession,
    ClientWebSocketResponse,
    WebSocketError,
)

from . import CommandFailure, WebsocketFailure
from .web_socket_containers import (
    AnovaCommand,
    APCWifiDevice,
    build_a3_payload,
    build_a6_a7_payload,
    build_wifi_cooker_state_body,
)

_LOGGER = logging.getLogger(__name__)

# Anova's server never sends anything when a device is idle, so without a
# heartbeat aiohttp has no way to detect a connection that has died silently
# (e.g. a NAT/router timeout) instead of via a clean close frame.
WEBSOCKET_HEARTBEAT_SECONDS = 30

# How long to wait for a RESPONSE to a sent command before raising CommandFailure.
COMMAND_TIMEOUT = 10

# Backoff for reconnecting after the websocket drops unexpectedly (heartbeat
# timeout, NAT/router reset, etc). Doubles on each failed attempt up to the
# max, and resets once a connection succeeds again.
RECONNECT_INITIAL_DELAY_SECONDS = 5
RECONNECT_MAX_DELAY_SECONDS = 300


class AnovaWebsocketHandler:
    def __init__(self, firebase_jwt: str, jwt: str, session: ClientSession):
        self._firebase_jwt = firebase_jwt
        self.jwt = jwt
        self.session = session
        self.url = f"https://devices.anovaculinary.io/?token={self._firebase_jwt}&supportedAccessories=APC&platform=android"  # noqa
        self.devices: dict[str, APCWifiDevice] = {}
        self.ws: ClientWebSocketResponse | None = None
        self._message_listener: Future[None] | None = None
        # Requests awaiting a matching RESPONSE message, keyed by requestId.
        self._pending_commands: dict[str, Future[None]] = {}
        # Set by disconnect() so the reconnect loop knows a dropped
        # connection was intentional and shouldn't be retried.
        self._closing = False

    async def connect(self) -> None:
        """Connect and start the supervising listener task.

        Raises WebsocketFailure if this *initial* connection attempt fails.
        Once connected, later drops are retried automatically in the
        background (see _run_with_reconnect) instead of raising.
        """
        self._closing = False
        await self._connect_once()
        self._message_listener = asyncio.ensure_future(self._run_with_reconnect())

    async def _connect_once(self) -> None:
        try:
            self.ws = await self.session.ws_connect(
                self.url, heartbeat=WEBSOCKET_HEARTBEAT_SECONDS
            )
        except (WebSocketError, ClientConnectionError, TimeoutError, OSError) as ex:
            raise WebsocketFailure("Failed to connect to the websocket") from ex

    async def disconnect(self) -> None:
        self._closing = True
        if self._message_listener is not None:
            self._message_listener.cancel()
        if self.ws is not None:
            await self.ws.close()

    async def _reconnect_with_backoff(self) -> None:
        """Retry _connect_once with growing backoff until it succeeds or
        disconnect() is called."""
        delay = RECONNECT_INITIAL_DELAY_SECONDS
        while not self._closing:
            try:
                await self._connect_once()
            except WebsocketFailure:
                _LOGGER.warning(
                    "Anova websocket reconnect attempt failed, retrying in %s seconds",
                    delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_MAX_DELAY_SECONDS)
                continue
            return

    async def _run_with_reconnect(self) -> None:
        """Consume messages until the connection drops, then reconnect with
        backoff and resume - instead of leaving the listener dead forever."""
        while not self._closing:
            try:
                await self.message_listener()
            except Exception:
                # A bad message must not kill reconnection - see docstring.
                _LOGGER.exception("Anova websocket listener crashed")

            if self._closing:
                return

            if self.ws is not None and not self.ws.closed:
                await self.ws.close()

            _LOGGER.warning("Anova websocket disconnected, reconnecting")
            await self._reconnect_with_backoff()
            if not self._closing:
                _LOGGER.info("Anova websocket reconnected")

    async def send_command(
        self, command: AnovaCommand, payload: dict[str, Any]
    ) -> None:
        """Send a command and wait for the device to acknowledge it via RESPONSE."""
        if self.ws is None:
            raise WebsocketFailure(
                "Cannot send a command, the websocket is not connected."
            )
        request_id = str(uuid.uuid4())
        future: Future[None] = asyncio.get_event_loop().create_future()
        self._pending_commands[request_id] = future
        try:
            await self.ws.send_json(
                {"command": command.value, "requestId": request_id, "payload": payload}
            )
            await asyncio.wait_for(future, timeout=COMMAND_TIMEOUT)
        except asyncio.TimeoutError as ex:
            raise CommandFailure(
                f"Timed out waiting for a response to {command.value}"
            ) from ex
        except ClientConnectionResetError as ex:
            raise WebsocketFailure(
                "Websocket connection was lost while sending a command"
            ) from ex
        finally:
            self._pending_commands.pop(request_id, None)

    def on_message(self, message: dict[str, Any]) -> None:
        _LOGGER.debug("Found message %s", message)
        if message["command"] == AnovaCommand.EVENT_APC_WIFI_LIST:
            payload = message["payload"]
            for device in payload:
                if device["cookerId"] not in self.devices:
                    self.devices[device["cookerId"]] = APCWifiDevice(
                        cooker_id=device["cookerId"],
                        type=device["type"],
                        paired_at=device["pairedAt"],
                        name=device["name"],
                        send_command=self.send_command,
                    )
        elif message["command"] == AnovaCommand.EVENT_APC_STATE:
            cooker_id = message["payload"]["cookerId"]
            device = self.devices.get(cooker_id)
            if device is None:
                return
            state = message["payload"]["state"]
            if "job" in state:
                update = build_wifi_cooker_state_body(state).to_apc_update()
            elif message["payload"]["type"] == "a3":
                merged_state = {**(device.last_raw_a3_state or {}), **state}
                device.last_raw_a3_state = merged_state
                update = build_a3_payload(merged_state)
            elif message["payload"]["type"] in {"a6", "a7"}:
                update = build_a6_a7_payload(state)
            else:
                return
            device.last_update = update
            device.last_update_received_at = datetime.now(UTC)
            if device.update_listener is not None:
                device.update_listener(update)
        elif message["command"] == AnovaCommand.RESPONSE:
            self._resolve_pending_command(message)

    def _resolve_pending_command(self, message: dict[str, Any]) -> None:
        request_id = message.get("requestId")
        future = (
            self._pending_commands.get(request_id) if request_id is not None else None
        )
        if future is None or future.done():
            return
        payload = message.get("payload") or {}
        if payload.get("status") == "ok":
            future.set_result(None)
        else:
            future.set_exception(CommandFailure(f"Command was rejected: {payload}"))

    async def message_listener(self) -> None:
        if self.ws is not None:
            async for msg in self.ws:
                self.on_message(json.loads(msg.data))
