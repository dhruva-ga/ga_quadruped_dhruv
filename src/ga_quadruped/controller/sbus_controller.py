import zmq, struct, time

FMT = "<13f"  # ch1..ch13, normalized [-1, 1]
PAYLOAD_SIZE = struct.calcsize(FMT)

class SbusVelocityController:
    """
    ZMQ/SBUS-driven velocity controller (event-in, no polling from keyboard).

    Channel mapping (1-indexed in YAML, 0-indexed in payload):
      ch2 (r_fwd_back)  -> w
      ch3 (l_fwd_back)  -> vx
      ch4 (l_left_right)-> vy
    """

    def __init__(
        self,
        endpoint="tcp://localhost:8011",
        topic=b"sbus",
        vmax_lin=1.0,
        vmax_ang=1.0,
        deadzone=0.03,
        invert_left_vertical=True,
        invert_right_vertical=True,
        conflate=True,
        rcvhwm=1,                # keep inbound queue tiny
        stale_after_s=0.0,       # 0=disabled. If >0, zero commands if no update for this long
    ):
        self.vmax_lin = float(vmax_lin)
        self.vmax_ang = float(vmax_ang)
        self.deadzone = float(deadzone)
        self.invert_left_vertical = bool(invert_left_vertical)
        self.invert_right_vertical = bool(invert_right_vertical)
        self.stale_after_s = float(stale_after_s)

        # state
        self.vx = 0.0
        self.vy = 0.0
        self.w  = 0.0
        self._last_msg_time = 0.0

        # ZMQ SUB socket
        ctx = zmq.Context.instance()
        self._sub = ctx.socket(zmq.SUB)
        if conflate:
            self._sub.setsockopt(zmq.CONFLATE, 1)
        if rcvhwm is not None:
            self._sub.setsockopt(zmq.RCVHWM, int(rcvhwm))
        self._sub.connect(endpoint)
        self._sub.setsockopt(zmq.SUBSCRIBE, topic)  # prefix match

    def _apply_payload(self, payload: bytes):
        if not payload or len(payload) != PAYLOAD_SIZE:
            return  # malformed; keep last commands
        vals = struct.unpack(FMT, payload)  # tuple of 13 floats

        # extract channels (0-based indices for ch1..ch13)
        r_left_right  = vals[0]  # ch1 (unused)
        r_fwd_back    = vals[1]  # ch2 -> w
        l_fwd_back    = vals[2]  # ch3 -> vx
        l_left_right  = vals[3]  # ch4 -> vy

        # optional axis inversion (common that "forward" = negative)
        if self.invert_left_vertical:
            l_fwd_back = -l_fwd_back
        if self.invert_right_vertical:
            r_fwd_back = -r_fwd_back

        # deadzone
        if abs(l_fwd_back)   < self.deadzone: l_fwd_back = 0.0
        if abs(l_left_right) < self.deadzone: l_left_right = 0.0
        if abs(r_fwd_back)   < self.deadzone: r_fwd_back = 0.0

        # scale to commanded velocities
        self.vx = float(l_fwd_back)   * self.vmax_lin
        self.vy = float(l_left_right) * self.vmax_lin
        self.w  = float(r_fwd_back)   * self.vmax_ang
        self._last_msg_time = time.time()

    def step(self, timeout_ms=0):
        """
        Pump at most one incoming SBUS message (non-blocking by default),
        update (vx, vy, w), and return them.
        """
        print("step called")
        # optional timeout wait
        if timeout_ms and self._sub.poll(timeout_ms) == 0:
            # no data within timeout
            if self.stale_after_s and (time.time() - self._last_msg_time) > self.stale_after_s:
                self.reset()
            return self.vx, self.vy, self.w

        # try to read whatever is there; with CONFLATE there’s at most one
        while True:
            try:
                frames = self._sub.recv_multipart(flags=zmq.DONTWAIT)
                print("frames", frames)
            except zmq.Again:
                break

            # Accept [topic, payload] or [topic, header, payload]
            if len(frames) == 2:
                _, payload = frames
            elif len(frames) >= 3:
                # take the last frame as payload; middle frames treated as headers/metadata
                payload = frames[-1]
            else:
                payload = b""

            print("payload", payload)
            self._apply_payload(payload)

            # If you’re NOT using CONFLATE and want to ensure "latest now", keep looping.
            # With CONFLATE, there’s at most one message queued, so we’ll break on Again.

        if self.stale_after_s and (time.time() - self._last_msg_time) > self.stale_after_s:
            self.reset()

        return self.vx, self.vy, self.w

    def get(self):
        return self.vx, self.vy, self.w

    def reset(self):
        self.vx = self.vy = self.w = 0.0

    def __repr__(self):
        return f"SbusVelocityController(vx={self.vx:.3f}, vy={self.vy:.3f}, w={self.w:.3f})"


if __name__ == "__main__":
    ctrl = SbusVelocityController(
        vmax_lin=1.0,   # m/s
        vmax_ang=1.0,   # rad/s
        deadzone=0.05,
        invert_left_vertical=False,
        invert_right_vertical=False,
        stale_after_s=0.0,  # e.g., set to 0.5 to zero commands if SBUS is silent for 0.5s
    )

    while True:
        vx, vy, w = ctrl.step(timeout_ms=1)  # non-blocking-ish
        print(vx, vy, w)
        time.sleep(0.02)
