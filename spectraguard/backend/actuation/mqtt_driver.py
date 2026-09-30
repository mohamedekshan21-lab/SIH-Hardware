"""MQTT Driver for physical reject actuator / sorter integration."""

from __future__ import annotations

import json
import logging
import time
from typing import Optional, Callable

logger = logging.getLogger(__name__)

try:
    import paho.mqtt.client as mqtt
    HAS_MQTT = True
except ImportError:
    HAS_MQTT = False
    mqtt = None


class MQTTDriver:
    """Sends reject commands over MQTT to an industrial PLC or sorter controller.

    Topics:
      - Publish: spectraguard/actuator/reject
      - Subscribe: spectraguard/actuator/ack
    """

    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        topic_reject: str = "spectraguard/actuator/reject",
        topic_ack: str = "spectraguard/actuator/ack",
    ):
        self.broker = broker
        self.port = port
        self.topic_reject = topic_reject
        self.topic_ack = topic_ack
        self._client: Optional[mqtt.Client] = None if HAS_MQTT else None
        self._connected = False
        self.pending_acks: dict[str, float] = {}

    def connect(self):
        if not HAS_MQTT:
            logger.warning("paho-mqtt not installed. MQTTDriver falling back to stub mode.")
            self._connected = False
            return

        try:
            self._client = mqtt.Client(client_id="spectraguard_actuator_service")
            self._client.on_connect = self._on_connect
            self._client.on_message = self._on_message
            self._client.connect(self.broker, self.port, keepalive=60)
            self._client.loop_start()
        except Exception as e:
            logger.error(f"MQTT connection failed to {self.broker}:{self.port}: {e}")
            self._connected = False

    def disconnect(self):
        if self._client and self._connected:
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"MQTT Connected to {self.broker}:{self.port}")
            self._connected = True
            client.subscribe(self.topic_ack)
        else:
            logger.error(f"MQTT Connect failed with code {rc}")
            self._connected = False

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            item_id = payload.get("item_id")
            if item_id:
                self.pending_acks[item_id] = time.time()
                logger.info(f"MQTT Ack received for item {item_id}")
        except Exception as e:
            logger.error(f"Failed to process ACK message: {e}")

    def fire_reject(self, item_id: str, delay_ms: float = 0) -> bool:
        if not self._connected or not self._client:
            logger.warning(f"MQTT not connected. Cannot fire reject for {item_id}")
            return False

        payload = {
            "item_id": item_id,
            "action": "REJECT",
            "timestamp": time.time(),
            "delay_ms": delay_ms,
        }
        self._client.publish(self.topic_reject, json.dumps(payload), qos=1)
        # Wait up to 100ms for ACK
        t0 = time.time()
        while time.time() - t0 < 0.1:
            if item_id in self.pending_acks:
                del self.pending_acks[item_id]
                return True
            time.sleep(0.005)

        return False

    def fire_hold(self, item_id: str) -> bool:
        if not self._connected or not self._client:
            return False

        payload = {
            "item_id": item_id,
            "action": "HOLD",
            "timestamp": time.time(),
        }
        self._client.publish(self.topic_reject, json.dumps(payload), qos=1)
        return True
