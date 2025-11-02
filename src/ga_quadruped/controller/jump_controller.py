from ga_quadruped.controller.controller_interface import ControlOutput, ControllerInterface, jump_spec


class JumpController(ControllerInterface):
    """
    Jump-only controller:
      - Space / 'j' triggers a jump window.
      - While active, events['jump'] == True for n_jump_steps frames.
      - axes['jump_phase'] = 1 - (steps_left / total_steps) in [0, 1), resets to 0 when inactive.
    """

    def __init__(self, n_jump_steps: int = 60):
        self._spec = jump_spec()
        self.n_jump_steps = max(1, int(n_jump_steps))  # avoid div-by-zero
        self._jump_steps_left = 0

    @property
    def spec(self):
        return self._spec

    def handle_event(self, key: str) -> None:
        if key in (" ", "j", "J"):
            # (Re)start jump window
            self._jump_steps_left = self.n_jump_steps

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
        
        print(f"JumpController.step(): jumping={jumping}, phase={phase:.3f}, steps_left={self._jump_steps_left}")
        return ControlOutput(self._spec, axes={"jump_phase": phase}, events={"jump": jumping})

    def get(self) -> ControlOutput:
        # Read-only snapshot (does not consume a step)
        return ControlOutput(self._spec, axes={"jump_phase": self._phase()}, events={"jump": self._jump_steps_left > 0})

    def reset(self) -> None:
        self._jump_steps_left = 0
