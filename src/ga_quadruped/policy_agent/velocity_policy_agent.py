import numpy as np

from ga_quadruped.controller.controller_interface import ControlOutput
from ga_quadruped.policy_agent.policy_agent import PolicyAgentInterface

def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    # qx = -qx  # Negate x component
    # qy = -qy  # Negate y component
    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation.astype(np.float32)
class VelocityPolicyAgent(PolicyAgentInterface):
    def __init__(self,
                 controller,
                 robot,
                 onnx_path: str,
                 default_qpos: np.ndarray,             # RENAMED
                 gait_freq: float = 1.25,
                 dt: float = 0.02,
                 action_scale: float = 0.25):
        super().__init__(controller, robot, onnx_path, default_qpos, action_scale=action_scale)
        self.command = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # [vx, vy, w]
        self.phase = np.array([0.0], dtype=np.float32)
        print('gait freq', gait_freq)
        self.phase_dt = 2.0 * np.pi * dt * float(gait_freq)
        self.action_scale = 0.25
        self.ang_vel_scale = 0.2
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.joint_ids = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
        self.sim_time = 0.0
        self.gait_freq = gait_freq  

    def consume_control(self, out: ControlOutput) -> None:
        if getattr(out, "axes", None):
            vx = out.axes.get("vx", self.command[0])
            vy = out.axes.get("vy", self.command[1])
            w  = out.axes.get("w",  self.command[2])
            self.command = np.array([vx, vy, w], dtype=np.float32)

    def compute_obs(self) -> np.ndarray:
        qpos, qvel, imu_quat, gyro = self._read_robot_signals()
        z_axis = get_gravity_orientation(imu_quat)#self.compute_gravity_orientation(imu_quat)
        cos, sin = np.cos(self.phase), np.sin(self.phase)
        #phase_feat = np.concatenate([sin, cos], axis=0).astype(np.float32)

        gait_phase = (self.sim_time * self.gait_freq) % 1.0
        phase_feat = np.array([
                np.sin(2 * np.pi * gait_phase),
                np.cos(2 * np.pi * gait_phase)
            ], dtype=np.float32)
        height_map = 0.1*np.ones(187, dtype=np.float32)
        self.sim_time += 0.02   
        z_axis[:2] = -z_axis[:2]  # Negate x and y components
        #gyro[:2] = -gyro[:2]
        # gyro[0] = -gyro[0]  
        # gyro[1] = -gyro[1]
        print('qvel', qvel.shape, qpos.shape, self.last_act)
        joint_pos_rel = (qpos - self.default_qpos) * self.dof_pos_scale
        state = np.concatenate([
            np.asarray(gyro, dtype=np.float32) * self.ang_vel_scale,
            -z_axis,
            self.command,                     # 3
            phase_feat, 
            np.array([2.0, 0.5, 0.5, 0.5, 0.0]),                                           # 8
            joint_pos_rel[self.joint_ids],       # RENAMED
            qvel[self.joint_ids]*self.dof_vel_scale,
             self.last_act,
             height_map  # padding to match original size
        ], axis=0)

        # state = np.concatenate([
        #     self.command,                     # 3
        #     phase_feat,                       # 8
        #     (qpos - self.default_qpos),       # RENAMED
        #     qvel,
        #     gyro,
        #     -z_axis,
        #     self.last_act,
        # ], axis=0)


        self.phase = ((self.phase + self.phase_dt) + np.pi) % (2.0 * np.pi) - np.pi
        return state.reshape(1, -1).astype(np.float32).clip(-5.0, 5.0)
