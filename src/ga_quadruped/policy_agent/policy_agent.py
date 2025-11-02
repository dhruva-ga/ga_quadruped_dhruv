from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import numpy as np
import onnxruntime as ort

from ga_quadruped.controller.controller_interface import ControlOutput, ControllerInterface
from ga_quadruped.robot.base_robot import BaseRobot

class PolicyAgentInterface(ABC):
    def __init__(
        self,
        controller: ControllerInterface,
        robot: BaseRobot,
        onnx_path: str,
        default_qpos: np.ndarray,  # RENAMED
        action_scale: float = 0.5,
    ):
        self.controller = controller
        self.robot = robot
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.default_qpos = default_qpos.copy()  # RENAMED
        self.last_act = default_qpos.copy()  # keep same init behavior
        self.action_scale = float(action_scale)
        self.last_signals: Dict[str, Any] = {}

    def pull_control(self, **kwargs) -> ControlOutput:
        if self.controller is None:
            return ControlOutput()
        
        return self.controller.step(**kwargs)

    @abstractmethod
    def consume_control(self, out: ControlOutput) -> None: ...

    def _read_robot_signals(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        qpos = self.robot.get_position().copy()
        qvel = self.robot.get_velocity().copy()
        imu_quat = self.robot.get_imu_quat()
        gyro = self.robot.get_gyro().copy()
        self.last_signals.update(qpos=qpos, qvel=qvel, imu_quat=imu_quat, gyro=gyro)
        return qpos, qvel, imu_quat, gyro

    @staticmethod
    def compute_gravity_orientation(quat: np.ndarray) -> np.ndarray:
        gx = 2 * (quat[1] * quat[3] - quat[0] * quat[2])
        gy = 2 * (quat[0] * quat[1] + quat[2] * quat[3])
        gz = (
            quat[0] * quat[0]
            - quat[1] * quat[1]
            - quat[2] * quat[2]
            + quat[3] * quat[3]
        )
        return np.array([-gx, -gy, gz], dtype=np.float32)

    @abstractmethod
    def compute_obs(self) -> np.ndarray: ...

    def act(self, obs: np.ndarray) -> np.ndarray:
        output = (
            self.session.run(None, {"input": obs.reshape(1, -1)})[0]
            .flatten()
            .astype(np.float32)
        )
        self.last_act = output.copy()
        return self.default_qpos + self.action_scale * output

    def tick(self, control_timeout_ms: int = 1):
        out = self.pull_control(timeout_ms=control_timeout_ms)
        self.consume_control(out)
        obs = self.compute_obs()
        ctrl = self.act(obs)
        self.robot.set_ctrl(ctrl)
        return out, obs, ctrl

    # Optional setter with new name
    def set_default_qpos(self, qpos: np.ndarray) -> None:
        self.default_qpos = qpos.copy()
