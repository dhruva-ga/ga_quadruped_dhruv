import numpy as np

from ga_quadruped.controller.controller_interface import ControlOutput
from ga_quadruped.policy_agent.policy_agent import PolicyAgentInterface

class JumpPolicyAgent(PolicyAgentInterface):
    def __init__(self,
                 controller,
                 robot,
                 onnx_path: str,
                 default_qpos: np.ndarray,            # RENAMED
                 action_scale: float = 0.5):
        super().__init__(controller, robot, onnx_path, default_qpos, action_scale=action_scale)
        self.is_jumping = False
        self.jump_command = 0.0
        self.jump_phase = 0.0

    def consume_control(self, out: ControlOutput) -> None:
        self.is_jumping = bool(out.events.get("jump"))
        self.jump_phase = out.axes.get("jump_phase")
        self.jump_command = out.axes.get("jump_height")

    def compute_obs(self) -> np.ndarray:
        qpos, qvel, imu_quat, gyro = self._read_robot_signals()
        z_axis = self.compute_gravity_orientation(imu_quat)
        header = np.array([float(self.is_jumping), float(self.jump_command), self.jump_phase], dtype=np.float32)

        state = np.concatenate([
            header,
            (qpos - self.default_qpos),       # RENAMED
            qvel,
            gyro,
            -z_axis,
            self.last_act,
        ], axis=0)

        return state.reshape(1, -1).astype(np.float32)
