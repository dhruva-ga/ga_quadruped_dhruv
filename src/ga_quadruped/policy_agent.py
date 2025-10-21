import time
import numpy as np
import onnxruntime as ort
import numpy as np
from scipy.spatial.transform import Rotation as R
np.set_printoptions(suppress=True,precision=3)
from scipy.signal import butter

class PolicyAgent:
    """
    Handles observation computation, command logic, and action inference.
    """
    def __init__(self,
                 onnx_path: str,
                 initial_qpos: np.ndarray):
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.last_act = np.array(initial_qpos.copy())
        self.initial_qpos = initial_qpos
        self.command = np.array([0.0, 0.0,0.0])  # forward, turn
        gait_freq = 2.0
        dt = 0.02
        self.phase_dt = 2 * np.pi * dt * gait_freq
        self.phase = np.array([0,0.5 * np.pi,np.pi,1.5 * np.pi])
        self.jump_command = 0.0

    def set_jump_command(self, jump_command: float):
        """Set the jump command for the robot."""
        self.jump_command = jump_command

    def set_initial_qpos(self, qpos: np.ndarray):
        """Set the initial position of the robot."""
        self.initial_qpos = qpos.copy()

    
    def set_command(self, command: np.ndarray):
        """Set the command for the robot."""
        self.command = command.copy()

    def compute_gravity_orientation(self, quat: np.ndarray) -> np.ndarray:
        gx = 2 * (quat[1] * quat[3] - quat[0] * quat[2])
        gy = 2 * (quat[0] * quat[1] + quat[2] * quat[3])
        gz = quat[0] * quat[0] - quat[1] * quat[1] - quat[2] * quat[2] + quat[3] * quat[3]
        return np.array([-gx, -gy, gz], dtype=np.float32)
    
    def compute_obs(self, is_jumping, qpos,qvel,imu_quat,gyro,grav,linvel) -> np.ndarray:
        z_axis = self.compute_gravity_orientation(imu_quat)

        cos = np.cos(self.phase)
        sin = np.sin(self.phase)
        phase = np.concatenate([cos, sin], axis=0).astype(np.float32)

        is_jumping = np.array([is_jumping])
        jump_command = np.array([self.jump_command])
        state = np.concatenate([
            # is_jumping,
            # jump_command,
            self.command,
            phase,
            #linvel,
            qpos - self.initial_qpos,
            qvel,
            gyro,
            -z_axis,
            self.last_act,
        ], axis=0)

        self.phase += self.phase_dt
        self.phase = np.mod(self.phase + np.pi, 2 * np.pi) - np.pi  # Normalize to [-pi, pi]

        return state.reshape(1, -1).astype(np.float32)
    

    def act(self, obs: np.ndarray) -> np.ndarray:
        output = self.session.run(None, {'input': obs.reshape(1,-1)})[0].flatten().astype(np.float32)
        self.last_act = output.copy()
        #action_scale_array = np.array([1.61161006,2.19426908,2.19426908,2.19426908,2.23729396,2.23729396,
        #                              1.61161006,2.19426908,2.19426908,2.19426908,2.23729396,2.23729396,])
        #scaled_output = action_scale_array * output
        action_scale = 0.5
        scaled_output = action_scale * output        
        real_ctrl = self.initial_qpos + scaled_output
        return real_ctrl