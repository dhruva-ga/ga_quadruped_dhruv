from ga_quadruped.controller.controller_interface import ControlOutput, ControlSpec, ControllerInterface

def jump_spec() -> ControlSpec:
    return ControlSpec(
        kind="jump",
        # Now exposes two continuous axes: jump_phase and jump_height
        axes=("jump_phase", "jump_height"),
        events=("jump", "quit"),
    )

class JumpController(ControllerInterface):
    """
    Jump-only controller:
      - Space / 'j' triggers a jump window.
      - While active, events['jump'] == True for n_jump_steps frames.
      - axes['jump_phase'] = 1 - (steps_left / total_steps) in [0, 1), resets to 0 when inactive.
      - axes['jump_height'] in [0.55, 1.2]; 'w' increases, 's' decreases (clamped).
    """

    MIN_JUMP = 0.6
    MAX_JUMP = 0.9

    def __init__(self, n_jump_steps: int = 60, height_step: float = 0.05):
        self._spec = jump_spec()
        self.n_jump_steps = max(1, int(n_jump_steps))  # avoid div-by-zero
        self._jump_steps_left = 0

        # Height control
        self._jump_height = 0.6
        self._height_step = float(height_step)

    @property
    def spec(self):
        return self._spec

    def handle_event(self, key: str) -> None:
        if key in (" ", "j", "J"):
            # (Re)start jump window
            self._jump_steps_left = self.n_jump_steps
        elif key in ("w", "W"):
            # Increase jump height (clamped)
            self._jump_height = min(self.MAX_JUMP, self._jump_height + self._height_step)
        elif key in ("s", "S"):
            # Decrease jump height (clamped)
            self._jump_height = max(self.MIN_JUMP, self._jump_height - self._height_step)

    def _phase(self) -> float:
        if self._jump_steps_left <= 0:
            return 0.0
        # 1 - (n_left / total)
        phase = 1.0 - (self._jump_steps_left / float(self.n_jump_steps))
        # Numerical safety clamp
        return 0.0 if phase < 0.0 else (1.0 if phase > 1.0 else phase)

    def step(self, **kwargs) -> ControlOutput:
        jumping = self._jump_steps_left > 0
        phase = self._phase()
        if jumping:
            self._jump_steps_left -= 1  # consume one frame

        print(
            f"JumpController: jumping={jumping}, "
            f"phase={phase:.3f}, steps_left={self._jump_steps_left}, "
            f"jump_height={self._jump_height:.3f}"
        )
        return ControlOutput(
            self._spec,
            axes={"jump_phase": phase, "jump_height": self._jump_height},
            events={"jump": jumping},
        )

    def get(self) -> ControlOutput:
        # Read-only snapshot (does not consume a step)
        return ControlOutput(
            self._spec,
            axes={"jump_phase": self._phase(), "jump_height": self._jump_height},
            events={"jump": self._jump_steps_left > 0},
        )

    def reset(self) -> None:
        self._jump_steps_left = 0
        # Intentionally keep the current height; remove the next line if you
        # prefer reset to also restore the default height.
        # self._jump_height = 0.6
