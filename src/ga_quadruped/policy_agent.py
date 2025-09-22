import time
import numpy as np
import onnxruntime as ort
import numpy as np
from scipy.spatial.transform import Rotation as R
np.set_printoptions(suppress=True,precision=3)

class PolicyAgent:
    """
    Handles observation computation, command logic, and action inference.
    """
    def __init__(self,
                 onnx_path: str,
                 initial_qpos: list):
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.command = np.array([0.0,0.0,0], dtype=np.float32)
        # self.rot_180 = R.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix()
        # self.rot_180_inv = np.linalg.inv(self.rot_180)
        self.action_scale = 0.25
        self.ang_vel_scale = 0.2
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.joint_ids = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
        # self.joint_ids = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        self.num_act_joints = 12

        self.initial_qpos = np.array(initial_qpos)
        # self.num_act_joints = len(self.joint_ids)
        self.last_act = np.zeros(self.num_act_joints, dtype=np.float32)
        # self.period = 0.8
        self.counter = 0
        self.dt = 0.005
        self.obs_length = 9 + self.num_act_joints * 3  # gyro + gravity orientation + command + qpos + qvel + last_act
        
        self.gait_command = np.array([1.5, 0.5, 0.5, 0.5, 0.0])
        # Test
        # self.fl_test_angles = np.linspace(initial_qpos[1], initial_qpos[1]+2.5, 200)
        # self.rr_test_angles = np.linspace(initial_qpos[10], initial_qpos[10]+2.5, 200)
        # self.test_counter = 0

    def set_gait_command(self, gait_command: np.ndarray):
        """Set the gait command vector for the agent."""
        if gait_command.shape != (5,):
            raise ValueError("Gait command must be a 5-element vector.")
        self.gait_command = gait_command.astype(np.float32)

    def set_command(self, command: np.ndarray):
        """Set the command vector for the agent."""
        if command.shape != (3,):
            raise ValueError("Command must be a 3-element vector.")
        self.command = command.astype(np.float32)
    
    # def set_initial_qpos(self, qpos: np.ndarray):
    #     """Set the initial position of the robot."""
    #     self.initial_qpos = qpos.copy()

    def compute_gravity_orientation(self, quat: np.ndarray) -> np.ndarray:
        gx = 2 * (quat[1] * quat[3] - quat[0] * quat[2])
        gy = 2 * (quat[0] * quat[1] + quat[2] * quat[3])
        gz = quat[0] * quat[0] - quat[1] * quat[1] - quat[2] * quat[2] + quat[3] * quat[3]
        return np.array([-gx, -gy, -gz], dtype=np.float32)


    def compute_obs(self, qpos, qvel, contact, imu_quat, accel, gyro, gait_phase) -> np.ndarray:

        z_axis = self.compute_gravity_orientation(imu_quat)
        # cos = np.cos(self.phase)
        # sin = np.sin(self.phase)
        # phase = np.concatenate([cos, sin], axis=0).astype(np.float32)

        # Policy space has different joint order
        qpos = (qpos - self.initial_qpos) * self.dof_pos_scale
        qpos = qpos[self.joint_ids]

        qvel = qvel * self.dof_vel_scale
        qvel = qvel[self.joint_ids]

        print("Z_axis: ",z_axis)
        state = np.concatenate([
            # gyro * self.ang_vel_scale,
            z_axis,
            self.command,
            gait_phase,
            self.gait_command,
            qpos,
            qvel,
            self.last_act,
        ], axis=0)
        #print(state.shape)

        # self.phase += self.phase_dt
        # self.phase = np.mod(self.phase + np.pi, 2 * np.pi) - np.pi  # Normalize to [-pi, pi]

        return state.reshape(1, -1).astype(np.float32)
        

    def act(self, obs: np.ndarray) -> np.ndarray:
        # 1) run the network with the correct input name
        inp = self.session.get_inputs()[0].name
        lab_act = self.session.run(None, {inp: obs})[0].flatten().astype(np.float32)

        # Clip the action
        lab_act = np.clip(lab_act, -4.0, 4.0)
        print("Clipping action between -4 and 4")

        # lab_act = np.zeros(12)
        # 2) stash for the next obs
        self.last_act = lab_act.copy()

        # 3) remap into mjc order
        mjc_act = np.zeros_like(lab_act)
        # mjc_act[:] = lab_act
        # mjc_act = np.array([0.0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        mjc_act[self.joint_ids] = lab_act
        # if self.test_counter < len(self.fl_test_angles):
        #     mjc_act = np.array([0, self.fl_test_angles[self.test_counter], -self.fl_test_angles[self.test_counter], 0, 0, 0, 0, 0, 0, 0, self.rr_test_angles[self.test_counter], -self.rr_test_angles[self.test_counter]])
        #     self.test_counter += 1
        # else:
        #     mjc_act = np.array([0, self.fl_test_angles[self.test_counter-1], -self.fl_test_angles[self.test_counter-1], 0, 0, 0, 0, 0, 0, 0, self.rr_test_angles[self.test_counter-1], -self.rr_test_angles[self.test_counter-1]])
        # 4) apply scale & add to default qpos
        print("Action: ", mjc_act)
        
        real_ctrl = self.initial_qpos + self.action_scale * mjc_act
        return real_ctrl.astype(np.float32)