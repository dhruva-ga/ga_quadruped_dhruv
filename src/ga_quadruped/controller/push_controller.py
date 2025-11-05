import warnings

from ga_quadruped.controller.controller_interface import (
    AxisSpec,
    ControlOutput,
    ControlSpec,
    ControllerInterface,
)

def push_spec() -> ControlSpec:
    return ControlSpec(
        kind="push",
        axes=(
            AxisSpec("mag", float, units="unitless"),
        ),
        events=("push", "quit"),
    )

class PushController(ControllerInterface):
    def __init__(
        self,
        mag_step=0.1,
        min_mag=0.1,
        max_mag=2.0,
        passthrough_keys=("q", "Q"),
    ):
        self._spec = push_spec()
        self.mag_step = float(mag_step)
        self.min_mag = float(min_mag)
        self.max_mag = float(max_mag)
        self.passthrough_keys = set(passthrough_keys)

        # state
        self.mag = self.min_mag
        self._push_latch = False

        # '[' decreases, ']' increases, 'p' triggers push
        self._keymap = {
            "[": -1,
            "]": +1,
            "p": "push",
        }

    @property
    def spec(self):
        return self._spec

    def step(self, **kwargs) -> ControlOutput:
        # Emit and clear the one-shot push event
        push_now = self._push_latch
        self._push_latch = False

        print(f"PushController: mag={self.mag:.2f}, push={push_now}")

        return ControlOutput(
            self._spec,
            axes={"mag": self.mag},
            events={"push": push_now, "quit": False},
        )

    def get(self) -> ControlOutput:
        return self.step()

    def reset(self) -> None:
        self.mag = self.min_mag
        self._push_latch = False

    def handle_event(self, key: str) -> None:
        if key == "t":
            self.reset()
            return

        action = self._keymap.get(key)
        if action == "push":
            self._push_latch = True
            return
        elif isinstance(action, int):
            # adjust magnitude
            new_mag = self.mag + action * self.mag_step
            self.mag = self._clip_range(new_mag, self.min_mag, self.max_mag)
            return

        if key not in self.passthrough_keys:
            warnings.warn(
                f"PushController: unrecognized key '{key}'", RuntimeWarning
            )

    @staticmethod
    def _clip_range(x: float, lo: float, hi: float) -> float:
        if lo > hi:
            # fallback: swap if misconfigured
            lo, hi = hi, lo
        return max(lo, min(hi, x))
