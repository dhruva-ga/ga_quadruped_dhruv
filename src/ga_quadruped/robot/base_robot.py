from abc import ABC, abstractmethod
import numpy as np


class BaseRobot(ABC):
    """Common interface for both sim and real robots used by the runner/policy.

    Implementations should provide joint-space signals (positions/velocities),
    IMU readings, and actuation, and should be safe to call at ~50 Hz.
    """

    # --- Control ---
    @abstractmethod
    def set_ctrl(self, ctrl: np.ndarray) -> None: ...

    @abstractmethod
    def get_ctrl(self) -> np.ndarray | None: ...

    # --- Joint state ---
    @abstractmethod
    def get_position(self) -> np.ndarray: ...   # shape (12,)

    @abstractmethod
    def get_velocity(self) -> np.ndarray: ...   # shape (12,)

    # --- IMU ---
    @abstractmethod
    def get_imu_quat(self) -> np.ndarray: ...   # scalar-first quaternion (w, x, y, z)

    @abstractmethod
    def get_gyro(self) -> np.ndarray: ...       # rad/s, shape (3,)

    # --- Telemetry (optional) ---
    def get_motor_torques(self) -> dict[str, float]:
        return {}
    
    def get_motor_temps(self) -> dict[str, float]:
        return {}
    
    # --- Lifecycle ---
    def step(self, nsteps: int = 1) -> None:
        """Advance simulation if needed. Real robots can no-op."""
        return None

    def start(self) -> None:
        """Start any background threads/loops if needed."""
        return None