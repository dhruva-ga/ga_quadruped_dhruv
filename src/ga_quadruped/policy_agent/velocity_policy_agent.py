import numpy as np

from ga_quadruped.controller.controller_interface import ControlOutput
from ga_quadruped.policy_agent.policy_agent import PolicyAgentInterface

class VelocityPolicyAgent(PolicyAgentInterface):
    def __init__(self,
                 controller,
                 robot,
                 onnx_path: str,
                 default_qpos: np.ndarray,             # RENAMED
                 gait_freq: float = 1.25,
                 dt: float = 0.02,
                 action_scale: float = 0.5):
        super().__init__(controller, robot, onnx_path, default_qpos, action_scale=action_scale)
        self.command = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # [vx, vy, w]
        self.phase = np.array([0.0, np.pi, np.pi, 0.0], dtype=np.float32)
        self.phase_dt = 2.0 * np.pi * dt * float(gait_freq)

    def consume_control(self, out: ControlOutput) -> None:
        if getattr(out, "axes", None):
            vx = out.axes.get("vx", self.command[0])
            vy = out.axes.get("vy", self.command[1])
            w  = out.axes.get("w",  self.command[2])
            self.command = np.array([vx, vy, w], dtype=np.float32)

    def compute_obs(self) -> np.ndarray:
        qpos, qvel, imu_quat, gyro = self._read_robot_signals()
        z_axis = self.compute_gravity_orientation(imu_quat)
        cos, sin = np.cos(self.phase), np.sin(self.phase)
        phase_feat = np.concatenate([cos, sin], axis=0).astype(np.float32)

        state = np.concatenate([
            self.command,                     # 3
            phase_feat,                       # 8
            (qpos - self.default_qpos),       # RENAMED
            qvel,
            gyro,
            -z_axis,
            self.last_act,
        ], axis=0)

        self.phase = ((self.phase + self.phase_dt) + np.pi) % (2.0 * np.pi) - np.pi
        return state.reshape(1, -1).astype(np.float32)
