import time
import numpy as np
from ga_quadruped.robot.base_robot import BaseRobot
from ga_quadruped.sim2sim.robot import SimRobot


class QuadrupedDefaultInitializer:
    def __init__(self,
                 robot: BaseRobot,
                 stand_gait: np.ndarray,
                 sit_gait: np.ndarray,
                 viewer=None
    ):
        self._stand_gait = stand_gait.copy()
        self._sit_gait = sit_gait.copy()
        self.robot = robot
        self.viewer = viewer

    # ----- Helpers -----
    def stand(self, seconds: float = 2.0) -> None:
        self._interp_to(self._stand_gait, seconds)

    def sit(self, seconds: float = 2.0) -> None:
        self._interp_to(self._sit_gait, seconds)

    def _interp_to(self, target: np.ndarray, seconds: float = 2.0) -> None:
        start = self.robot.get_position()
        steps = max(int(50 * seconds), 1)
        for i in range(steps):
            rate = (i + 1) / steps
            des = start * (1 - rate) + target * rate
            self.robot.set_ctrl(des)
            self.robot.step(nsteps=4)
            if self.viewer is not None:
                self.viewer.sync()
            time.sleep(0.01)


class QuadrupedDropInInitializer(QuadrupedDefaultInitializer):
    def __init__(self,
                 robot: SimRobot,
                 stand_gait: np.ndarray,
                 sit_gait: np.ndarray,
                 drop_height: float = 0.5,
                 viewer=None
    ):
        super().__init__(robot, stand_gait, sit_gait, viewer)
        self._drop_height = drop_height

    def drop_in(self, seconds: float = 2.0) -> None:
        # Set initial position higher
        self.robot.data.qpos[2] += self._drop_height
        self.robot.data.qpos[7:] = self._stand_gait