import asyncio
from copy import deepcopy
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientConnectionResetError

from anova_wifi import websocket_handler as websocket_handler_module
from anova_wifi.exceptions import WebsocketFailure
from anova_wifi.web_socket_containers import AnovaCommand, APCWifiDevice
from anova_wifi.websocket_handler import (
    WEBSOCKET_HEARTBEAT_SECONDS,
    AnovaWebsocketHandler,
)
from tests.example_data import A3_DELTA_MESSAGE, A3_IDLE_MESSAGE


class _FakeWebSocket:
    """A minimal stand-in for ClientWebSocketResponse: async-iterates over a
    fixed list of messages, then behaves as if the server closed the
    connection (matching what happens when a device drops off)."""

    def __init__(self, messages: list | None = None) -> None:
        self._messages = list(messages) if messages else []
        self.closed = False

    def __aiter__(self) -> "_FakeWebSocket":
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_connect_passes_heartbeat_to_ws_connect() -> None:
    """Without a heartbeat, aiohttp can't detect a silently dead connection."""
    session = AsyncMock()
    handler = AnovaWebsocketHandler(
        firebase_jwt="firebase_jwt", jwt="jwt", session=session
    )

    await handler.connect()

    session.ws_connect.assert_awaited_once_with(
        handler.url, heartbeat=WEBSOCKET_HEARTBEAT_SECONDS
    )

    await handler.disconnect()


@pytest.mark.asyncio
async def test_reconnects_after_websocket_drops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the connection dies (heartbeat timeout, NAT/router reset, ...),
    the listener must not just die - it should reconnect so data starts
    flowing again once the device comes back, instead of leaving every
    entity stuck unavailable until HA is restarted."""
    monkeypatch.setattr(websocket_handler_module, "RECONNECT_INITIAL_DELAY_SECONDS", 0)
    session = AsyncMock()
    first_ws = _FakeWebSocket()
    second_ws = _FakeWebSocket()
    session.ws_connect.side_effect = [first_ws, second_ws]

    handler = AnovaWebsocketHandler(
        firebase_jwt="firebase_jwt", jwt="jwt", session=session
    )
    await handler.connect()

    for _ in range(50):
        if session.ws_connect.await_count >= 2:
            break
        await asyncio.sleep(0)

    assert session.ws_connect.await_count == 2
    assert first_ws.closed is True
    assert handler.ws is second_ws

    await handler.disconnect()


@pytest.mark.asyncio
async def test_stops_reconnecting_after_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deliberate disconnect() must not trigger another reconnect attempt."""
    monkeypatch.setattr(websocket_handler_module, "RECONNECT_INITIAL_DELAY_SECONDS", 0)
    session = AsyncMock()
    session.ws_connect.side_effect = [_FakeWebSocket(), _FakeWebSocket()]

    handler = AnovaWebsocketHandler(
        firebase_jwt="firebase_jwt", jwt="jwt", session=session
    )
    await handler.connect()
    await handler.disconnect()

    for _ in range(10):
        await asyncio.sleep(0)

    assert session.ws_connect.await_count == 1


@pytest.mark.asyncio
async def test_send_command_wraps_connection_reset_as_websocket_failure() -> None:
    """A dying transport races the reconnect logic - send_json can raise
    ClientConnectionResetError even though self.ws is still set. Callers
    only handle WebsocketFailure/CommandFailure, so the raw aiohttp
    exception must be wrapped rather than propagated as-is."""
    handler = AnovaWebsocketHandler(
        firebase_jwt="firebase_jwt", jwt="jwt", session=AsyncMock()
    )
    handler.ws = AsyncMock()
    handler.ws.send_json.side_effect = ClientConnectionResetError(
        "Cannot write to closing transport"
    )

    with pytest.raises(WebsocketFailure):
        await handler.send_command(AnovaCommand.CMD_APC_STOP, {})


def test_state_push_caches_last_update_without_a_listener() -> None:
    """APCWifiDevice.available_commands needs last_update even with no
    update_listener attached, e.g. before HA has finished setting one up."""
    handler = AnovaWebsocketHandler(
        firebase_jwt="firebase_jwt", jwt="jwt", session=AsyncMock()
    )
    device = APCWifiDevice(cooker_id="x", type="pro", paired_at="now", name="test")
    handler.devices["x"] = device
    initial_last_update_received_at = device.last_update_received_at
    assert device.update_listener is None
    assert initial_last_update_received_at is None

    before = datetime.now(UTC)
    handler.on_message(
        {
            "command": "EVENT_APC_STATE",
            "payload": {
                "cookerId": "x",
                "type": "pro",
                "state": {
                    "job": {
                        "cook-time-seconds": 0,
                        "id": "job-id",
                        "mode": "COOK",
                        "ota-url": "",
                        "target-temperature": 60,
                        "temperature-unit": "C",
                    },
                    "job-status": {
                        "cook-time-remaining": 0,
                        "job-start-systick": 0,
                        "provisioning-pairing-code": 0,
                        "state": "COOKING",
                        "state-change-systick": 0,
                    },
                    "pin-info": {},
                    "temperature-info": {"water-temperature": 25.0},
                },
            },
        }
    )

    assert device.last_update is not None
    assert device.is_cooking is True
    assert device.last_update_received_at is not None
    assert before <= device.last_update_received_at
    assert device.last_update_received_at <= datetime.now(UTC)


def test_a3_delta_update_merges_onto_last_known_state() -> None:
    """A3 devices send a full state snapshot once, then omit unchanged
    top-level keys on later pushes. Without merging each push onto the
    last known state, a delta update would read those omitted fields as
    unset instead of retaining their last known value. Verified against
    real EVENT_APC_STATE traffic from an Anova Precision Cooker A3."""
    handler = AnovaWebsocketHandler(
        firebase_jwt="firebase_jwt", jwt="jwt", session=AsyncMock()
    )
    device = APCWifiDevice(
        cooker_id="anova random-id", type="a3", paired_at="now", name="test"
    )
    handler.devices["anova random-id"] = device

    full_snapshot = deepcopy(A3_IDLE_MESSAGE)
    full_snapshot["payload"]["state"]["timerInSeconds"] = 840
    handler.on_message(full_snapshot)
    assert device.last_update is not None
    assert device.last_update.sensor.cook_time_remaining == 840

    # A3_DELTA_MESSAGE omits timerInSeconds entirely.
    handler.on_message(deepcopy(A3_DELTA_MESSAGE))

    assert device.last_update is not None
    assert device.last_update.sensor.cook_time_remaining == 840
