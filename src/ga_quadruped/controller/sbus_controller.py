import zmq, struct, time

FMT = "<13f"  # ch1..ch13, normalized [-1, 1]
PAYLOAD_SIZE = struct.calcsize(FMT)

class SbusVelocityController:
    """
    ZMQ/SBUS-driven velocity controller (event-in, no polling from keyboard).

    Channel mapping (1-indexed in YAML, 0-indexed in payload):
      ch1 (r_left_right)  -> w   (angular)
      ch3 (l_fwd_back)    -> vx  (forward/back)
      ch4 (l_left_right)  -> vy  (left/right)
    """

    def __init__(
        self,
        endpoint="tcp://localhost:8011",
        topic=b"sbus",
        vmax_lin_x=1.0,
        vmax_lin_y=1.0,
        vmax_ang=1.0,
        deadzone=0.03,
        invert_left_vertical=True,
        invert_right_vertical=True,
        invert_left_left_right=False,   # invert left stick's horizontal axis (vy)
        invert_right_left_right=False,  # NEW: invert right stick's horizontal axis (w)
        button_threshold=0.5,           # > threshold => "pressed"
    ):
        self.vmax_lin_x = float(vmax_lin_x)
        self.vmax_lin_y = float(vmax_lin_y)
        self.vmax_ang = float(vmax_ang)
        self.button_threshold = float(button_threshold)

        self._quit_latch = False

        self.deadzone = float(deadzone)

        # Axis inversion options
        self.invert_left_vertical = bool(invert_left_vertical)
        self.invert_right_vertical = bool(invert_right_vertical)  # kept for future use
        self.invert_left_left_right = bool(invert_left_left_right)
        self.invert_right_left_right = bool(invert_right_left_right)  # NEW

        # state
        self.vx = 0.0
        self.vy = 0.0
        self.w  = 0.0
        self._last_msg_time = 0.0

        # ZMQ SUB socket
        ctx = zmq.Context.instance()
        self._sub = ctx.socket(zmq.SUB)
        # if conflate:
        #     self._sub.setsockopt(zmq.CONFLATE, 1)
        self._sub.connect(endpoint)
        self._sub.setsockopt(zmq.SUBSCRIBE, topic)  # prefix match

    def _apply_payload(self, payload: bytes):
        if not payload or len(payload) != PAYLOAD_SIZE:
            return  # malformed; keep last commands
        vals = struct.unpack(FMT, payload)  # tuple of 13 floats

        # extract channels (0-based indices for ch1..ch13)
        r_left_right  = vals[0]  # ch1 -> w
        r_fwd_back    = vals[1]  # ch2 (unused now)
        l_fwd_back    = vals[2]  # ch3 -> vx
        l_left_right  = vals[3]  # ch4 -> vy
        push_c        = vals[9]  # ch10 -> quit button (-1 idle, +1 pressed)

        # optional axis inversion
        if self.invert_left_vertical:
            l_fwd_back = -l_fwd_back
        if self.invert_right_vertical:
            r_fwd_back = -r_fwd_back  # kept in case ch2 used later
        if self.invert_left_left_right:
            l_left_right = -l_left_right
        if self.invert_right_left_right:      # NEW: invert right horizontal (affects w)
            r_left_right = -r_left_right

        # deadzone
        if abs(l_fwd_back)   < self.deadzone: l_fwd_back = 0.0
        if abs(l_left_right) < self.deadzone: l_left_right = 0.0
        if abs(r_left_right) < self.deadzone: r_left_right = 0.0

        # scale to commanded velocities
        self.vx = float(l_fwd_back)   * self.vmax_lin_x
        self.vy = float(l_left_right) * self.vmax_lin_y
        self.w  = float(r_left_right) * self.vmax_ang  # angular from ch1

        pressed = (push_c > self.button_threshold)
        if pressed:
            # rising edge detected: latch quit for one step()
            self._quit_latch = True

        self._last_msg_time = time.time()

    def step(self, timeout_ms=0):
        """
        Pump at most one incoming SBUS message (non-blocking by default),
        update (vx, vy, w), and return them.
        """
        while True:
            try:
                frames = self._sub.recv_multipart(flags=zmq.DONTWAIT)
            except zmq.Again:
                break

            # Accept [topic, payload] or [topic, header, payload]
            if len(frames) == 2:
                _, payload = frames
            elif len(frames) >= 3:
                payload = frames[-1]
            else:
                payload = b""

            self._apply_payload(payload)

        return self.vx, self.vy, self.w, self._quit_latch

    def get(self):
        return self.vx, self.vy, self.w

    def reset(self):
        self.vx = self.vy = self.w = 0.0

    def __repr__(self):
        return f"SbusVelocityController(vx={self.vx:.3f}, vy={self.vy:.3f}, w={self.w:.3f})"


if __name__ == "__main__":
    ctrl = SbusVelocityController(
        vmax_lin_x=1.0,   # m/s
        vmax_ang=1.0,   # rad/s
        deadzone=0.05,
        invert_left_vertical=False,
        invert_right_vertical=False,
        invert_left_left_right=True,
        invert_right_left_right=True,  # set True if right horizontal feels flipped
    )

    while True:
        vx, vy, w = ctrl.step(timeout_ms=0)  # non-blocking-ish
        print(vx, vy, w)
        time.sleep(0.02)
