import warnings

from ga_quadruped.controller.controller_interface import (
    AxisSpec,
    ControlOutput,
    ControlSpec,
    ControllerInterface,
    velocity_spec,
)

class VelocityController(ControllerInterface):
    def __init__(
        self,
        vel_step=0.05,
        max_lin_x=None,
        max_lin_y=None,
        max_ang=None,
        passthrough_keys=("q", "Q"),
    ):
        self._spec = velocity_spec()
        self.vel_step = float(vel_step)
        self.max_lin_x = None if max_lin_x is None else float(max_lin_x)
        self.max_lin_y = None if max_lin_y is None else float(max_lin_y)
        self.max_ang = None if max_ang is None else float(max_ang)
        self.passthrough_keys = set(passthrough_keys)
        self.vx = self.vy = self.w = 0.0
        self._keymap = {
            "w": ("vx", +1),
            "s": ("vx", -1),
            "a": ("vy", +1),
            "d": ("vy", -1),
            "g": ("w", +1),
            "h": ("w", -1),
        }

    @property
    def spec(self):
        return self._spec

    def step(self, **kwargs) -> ControlOutput:
        return ControlOutput(
            self._spec,
            axes={"vx": self.vx, "vy": self.vy, "w": self.w},
            events={"quit": False},
        )

    def get(self) -> ControlOutput:
        return self.step()

    def reset(self) -> None:
        self.vx = self.vy = self.w = 0.0

    def handle_event(self, key: str) -> None:
        if key == "t":
            self.reset()
            return
        target = self._keymap.get(key)
        if target:
            attr, sign = target
            val = getattr(self, attr) + sign * self.vel_step
            if attr == "vx":
                self.vx = self._clip(val, self.max_lin_x)
            elif attr == "vy":
                self.vy = self._clip(val, self.max_lin_y)
            elif attr == "w":
                self.w = self._clip(val, self.max_ang)
        else:
            if key not in self.passthrough_keys:
                warnings.warn(
                    f"VelocityController: unrecognized key '{key}'", RuntimeWarning
                )

    @staticmethod
    def _clip(x, lim):
        if lim is None:
            return x
        return max(-lim, min(lim, x))
