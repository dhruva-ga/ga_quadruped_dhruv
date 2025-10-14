import json
from typing import Any, Dict, Optional

import zmq
import numpy as np

class SimpleZmqPublisher:
    """Very small ZeroMQ PUB wrapper bound to tcp://*:9872.

    Usage:
        pub = SimpleZmqPublisher()        # binds to :9872
        pub.send({"a": 1}, topic="t1")
        pub.close()
    """

    def __init__(self, port: int = 9872) -> None:
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.bind(f"tcp://*:{int(port)}")

    def send(self, data: Dict[str, Any]) -> None:
        """Send a Python dict as JSON. If topic is provided, sends multipart [topic, payload]."""
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        try:
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Could not serialize data to JSON: {e}") from e
        self._socket.send(payload)

    def close(self) -> None:
        """Close socket and terminate context."""
        try:
            self._socket.close()
        finally:
            self._ctx.term()

    # convenience context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

if __name__ == "__main__":
    pub = SimpleZmqPublisher()
    counter = 0
    angular_freq = 0.5
    while True:
        sin_data = np.sin(counter * angular_freq)
        cos_data = np.cos(counter * angular_freq)
        pub.send({"counter_sin": sin_data, "counter_cos": cos_data, "counter": counter})
        counter += 1
        import time
        time.sleep(1)