import numpy as np

from ga_quadruped.controller.controller_interface import ControlOutput
from ga_quadruped.policy_agent.policy_agent import PolicyAgentInterface

class RecoveryPolicyAgent(PolicyAgentInterface):
    def __init__(self,
                 controller,
                 robot,
                 onnx_path: str,
                 default_qpos: np.ndarray,            # RENAMED
                 action_scale: float = 0.5):
        super().__init__(controller, robot, onnx_path, default_qpos, action_scale=action_scale)

    def consume_control(self, out: ControlOutput) -> None:
        pass

    def compute_obs(self) -> np.ndarray:
        qpos, qvel, imu_quat, gyro = self._read_robot_signals()
        z_axis = self.compute_gravity_orientation(imu_quat)
        state = np.concatenate([
            (qpos - self.default_qpos),       # RENAMED
            qvel,
            gyro,
            -z_axis,
            self.last_act,
        ], axis=0)
        return state.reshape(1, -1).astype(np.float32)
