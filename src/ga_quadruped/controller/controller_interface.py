from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


@dataclass(frozen=True)
class AxisSpec:
    name: str
    dtype: type = float  # float|int|bool (bool allowed for toggles-as-state)
    min: Optional[float] = None
    max: Optional[float] = None
    units: Optional[str] = None


@dataclass(frozen=True)
class ControlSpec:
    """
    Describes the action space for a controller.
    - kind: a short string like 'velocity', 'jump', 'pose', ...
    - axes: continuous/discrete values available every step (e.g., vx, vy, w)
    - events: one-shot triggers (latched true for a single step), e.g., 'quit', 'jump'
    """

    kind: str
    axes: Tuple[AxisSpec, ...] = ()
    events: Tuple[str, ...] = ()


# Convenience makers
def velocity_spec() -> ControlSpec:
    return ControlSpec(
        kind="velocity",
        axes=(
            AxisSpec("vx", float, units="m/s"),
            AxisSpec("vy", float, units="m/s"),
            AxisSpec("w", float, units="rad/s"),
        ),
        events=("quit",),
    )





# ---------- Output ----------


@dataclass(frozen=True)
class ControlOutput:
    """
    axes: current named axis values (use 0/False if not present)
    events: one-shot triggers latched for *this* step only
    """

    spec: ControlSpec = None
    axes: Dict[str, float] = field(default_factory=dict)
    events: Dict[str, bool] = field(default_factory=dict)

    def get_axis(self, name: str, default: float = 0.0) -> float:
        return float(self.axes.get(name, default))

    def fired(self, event: str) -> bool:
        return bool(self.events.get(event, False))


# ---------- Interface ----------


class ControllerInterface(ABC):
    @property
    @abstractmethod
    def spec(self) -> ControlSpec: ...

    @abstractmethod
    def step(self, **kwargs) -> ControlOutput:
        """Non-blocking update + return current ControlOutput."""
        ...

    @abstractmethod
    def get(self) -> ControlOutput:
        """Pure read (no I/O)."""
        ...

    @abstractmethod
    def reset(self) -> None: ...

    # Optional: event-driven input (e.g., keyboard key, button)
    def handle_event(self, event) -> None:
        raise NotImplementedError
