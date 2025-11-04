import zmq, struct, time

from ga_quadruped.controller.controller_interface import (
    ControlOutput,
    ControllerInterface,
    velocity_spec,
)

FMT = "<13f"
PAYLOAD_SIZE = struct.calcsize(FMT)


class SbusVelocityController(ControllerInterface):
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
        invert_left_left_right=False,
        invert_right_left_right=False,
        button_threshold=0.5,
        conflate=True,
    ):
        self._spec = velocity_spec()
        self.vmax_lin_x = float(vmax_lin_x)
        self.vmax_lin_y = float(vmax_lin_y)
        self.vmax_ang = float(vmax_ang)
        self.deadzone = float(deadzone)
        self.button_threshold = float(button_threshold)
        self.inv_lv = bool(invert_left_vertical)
        self.inv_rv = bool(invert_right_vertical)
        self.inv_llr = bool(invert_left_left_right)
        self.inv_rlr = bool(invert_right_left_right)
        self.vx = self.vy = self.w = 0.0
        self._quit_latch = False
        self._last_msg_time = 0.0
        ctx = zmq.Context.instance()
        self._sub = ctx.socket(zmq.SUB)
        if conflate:
            self._sub.setsockopt(zmq.CONFLATE, 1)
        self._sub.connect(endpoint)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")

    @property
    def spec(self):
        return self._spec

    def step(self, **kwargs) -> ControlOutput:
        poller = getattr(self, "_poller", None)
        if poller is None:
            poller = zmq.Poller()
            poller.register(self._sub, zmq.POLLIN)
            self._poller = poller
        socks = dict(poller.poll(20))  # ~1 ms; tune as needed
        if self._sub in socks:
            try:
                frames = self._sub.recv_multipart()  # safe after poll
                payload = frames[0]
                self._apply_payload(payload)
            except Exception as e:
                pass
        #try:
        #    frames = self._sub.recv_multipart(flags=zmq.DONTWAIT)
        #except zmq.Again:
        #    frames = None
        #if frames:
        #    payload = frames[-1] if len(frames) >= 2 else b""
        #    self._apply_payload(payload)
        #    print("Frames",self.vx,self.vy,self.w)
        quit_now = self._quit_latch
        return ControlOutput(
            self._spec,
            axes={"vx": self.vx, "vy": self.vy, "w": self.w},
            events={"quit": quit_now},
        )

    def get(self) -> ControlOutput:
        return ControlOutput(
            self._spec,
            axes={"vx": self.vx, "vy": self.vy, "w": self.w},
            events={"quit": False},
        )

    def reset(self) -> None:
        self.vx = self.vy = self.w = 0.0

    def _apply_payload(self, payload: bytes):
        if not payload or len(payload) != PAYLOAD_SIZE:
            return
        vals = struct.unpack(FMT, payload)
        r_lr, r_fb, l_fb, l_lr = vals[0], vals[1], vals[2], vals[3]
        push_c = vals[9]
        if self.inv_lv:
            l_fb = -l_fb
        if self.inv_rv:
            r_fb = -r_fb  # reserved
        if self.inv_llr:
            l_lr = -l_lr
        if self.inv_rlr:
            r_lr = -r_lr
        if abs(l_fb) < self.deadzone:
            l_fb = 0.0
        if abs(l_lr) < self.deadzone:
            l_lr = 0.0
        if abs(r_lr) < self.deadzone:
            r_lr = 0.0
        self.vx = float(l_fb) * self.vmax_lin_x
        self.vy = float(l_lr) * self.vmax_lin_y
        self.w = float(r_lr) * self.vmax_ang
        if push_c > self.button_threshold:
            self._quit_latch = True
        self._last_msg_time = time.time()

    def __repr__(self):
        return f"SbusVelocityController(vx={self.vx:.3f}, vy={self.vy:.3f}, w={self.w:.3f})"


if __name__ == "__main__":
    ctrl = SbusVelocityController(
        vmax_lin_x=1.0,  # m/s
        vmax_ang=1.0,  # rad/s
        deadzone=0.05,
        invert_left_vertical=False,
        invert_right_vertical=False,
        invert_left_left_right=True,
        invert_right_left_right=True,  # set True if right horizontal feels flipped
    )

    while True:
        ctrl.step(timeout_ms=0)  # non-blocking-ish
        print(ctrl.vx, ctrl.vy, ctrl.w)
        time.sleep(0.02)
