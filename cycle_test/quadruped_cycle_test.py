import os
from pathlib import Path

# Third Party
import numpy as np
import mujoco
import mujoco.viewer
import time
from scipy.spatial.transform import Rotation as R

# UV Imports
from quadruped_leg_ik.ik_client import IKSolverClient
from quadruped_leg_kinematics.param_helper import ParamLeg


class LowPassFilter:
    """Simple exponential moving average low pass filter for pose data"""
    def __init__(self, alpha=0.1):
        """
        Initialize low pass filter
        Args:
            alpha: smoothing factor (0 < alpha <= 1)
                  - Lower values = more smoothing (slower response)
                  - Higher values = less smoothing (faster response)
        """
        self.alpha = alpha
        self.prev_poses = None
        print(f"Initialized pose filter with alpha={self.alpha}")
        
    def filter(self, poses):
        """Apply low pass filter to poses"""
        if self.prev_poses is None:
            # First iteration - no filtering
            self.prev_poses = poses.copy()
            return poses
        
        # Apply exponential moving average
        filtered_poses = self.alpha * poses + (1 - self.alpha) * self.prev_poses
        self.prev_poses = filtered_poses.copy()
        return filtered_poses

class LegTeleop:
    def __init__(self, filter_alpha=0.1):        
        # Initialize low pass filter for poses
        self.pose_filter = LowPassFilter(alpha=filter_alpha)

        # Initialize IK solver client
        self.ik_client = IKSolverClient(ip="localhost", port=8120)

        self.xml_path = str(Path(__file__).parent / "assets" / "quadruped_single_leg.xml")

        # Initialize home joint angles
        self.prev_jt_angles = np.vstack([[0, 0, 0, 0], [ 0, 0, 0, 0]]) # Leg 
        self.home_jt_angles = np.vstack([[0, 0.0, 0.0, 0], [ 0, 0, 0, 0]]) # Leg 
        
        self.home_mat = self.solve_fk(self.home_jt_angles)

        # Initialize Param interface
        # self.param = ParamLeg()
        # self.param.start()
        # time.sleep(2)
        # self.kinematics_data = self.param.get_kinematics_data()

    def solve_ik(self, prev_jnt_angles: np.ndarray, poses: np.ndarray) -> np.ndarray:
        # compute IK
        sols = self.ik_client.solve(prev_jnt_angles, poses)
        return sols
    
    def solve_fk(self, angles: np.ndarray) -> np.ndarray:
        # Compute FK
        mats = self.ik_client.fk(angles)
        return mats

    def run_teleop(self):
        """Run the teleoperation simulation and sim2real with Mujoco viewer"""
        model = mujoco.MjModel.from_xml_path(self.xml_path)
        data = mujoco.MjData(model)

        # Launch viewer
        viewer = mujoco.viewer.launch_passive(model, data)

        t = 0
        while viewer.is_running(): 
            t+= 0.05
            # Generate raw poses
            raw_poses = self._generate_pos(t)
            
            # Apply low pass filter to poses
            filtered_poses = self.pose_filter.filter(raw_poses)
            sols = self.solve_ik(self.prev_jt_angles, filtered_poses)
            # print(sols)
            for i in range(model.nq):
                data.qpos[i] = sols[0][i]
                # data.qpos[i] = self.home_jt_angles[0][i]
                # Set control signals for actuators
                # self.param.set_ctrl(np.flip(sols, axis=0).flatten())
                # self.prev_jt_angles = sols.copy()
                print(self._check_xpos(model, data, "RR_foot"))
                print(self.solve_fk(sols))

                mujoco.mj_forward(model, data) 
                viewer.sync()

        print("Exiting Mujoco viewer")
        self.move_to_home()

    # Test Function
    def test_run_mujoco(self):
        model = mujoco.MjModel.from_xml_path(self.xml_path)
        data = mujoco.MjData(model)

        # Launch viewer
        viewer = mujoco.viewer.launch_passive(model, data)

        self.fk_results = []
        pose = np.tile(np.eye(4, dtype=np.float32)[None], (2, 1, 1))
        pose = self._get_mat(np.array([[0.706, -0.031, -0.706, -0.031, 0.314, -0.35,  0.1], [0.707,  0.,   -0.707,  0.,  0.314,  0.35,  0.1]]))
        while viewer.is_running():

            sols = self.solve_ik(self.prev_jt_angles, pose)
            for i in range(model.nq-6):
                data.qpos[i] = sols[0][i]
                data.qpos[i + model.nq - 6] = sols[1][i]
                # self.ga_one.set_ctrl(np.flip(sols, axis=0).flatten())
                mujoco.mj_forward(model, data) 
                fk_mat = self.solve_fk(sols)
                print(self.quat_wxyz_to_euler(fk_mat[0], degrees=True))
                print(self.quat_wxyz_to_euler(fk_mat[1], degrees=True))
                print(self.euler_deg_to_quat_wxyz(np.array([0,-90,-5])))
                print(f"FK Matrices:\n{fk_mat}")
                self.prev_jt_angles = sols.copy()
                viewer.sync()

    def move_to_home(self):
        """Move the arms to the home position"""
        ctrl = np.flip(self.home_jt_angles, axis = 0).flatten()
        self.ga_one.set_ctrl(ctrl)
        print("Reached home position")
        while True:
            time.sleep(0.1)

    # Helper Functions
    def _get_pos_quat(self, mats: np.ndarray):
        positions = mats[..., :3, 3]
        rot = R.from_matrix(mats[..., :3, :3].reshape(-1, 3, 3))
        xyzw = rot.as_quat()  # returns (x,y,z,w)
        quats = np.stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]], axis=1)
        batch = mats.shape[0]
        return positions.reshape(batch, 3), quats.reshape(batch, 4)
    
    def _get_mat(self, quatpos: np.ndarray):
        """
        Converts an array of quaternion-position vectors to homogeneous transformation matrices.

        Parameters
        ----------
        quatpos : np.ndarray
            An array of shape (N, 7), where each row contains a quaternion (w, x, y, z) followed by a position (x, y, z).

        Returns
        -------
        np.ndarray
            An array of shape (N, 4, 4), where each element is a 4x4 homogeneous transformation matrix representing the rotation and translation specified by the corresponding quaternion and position.
        """
        """"""
        mats = np.eye(4, dtype=np.float32)[None].repeat(quatpos.shape[0], axis=0)
        mats[:, :3, 3] = quatpos[:, 4:7]  # set positions
        r = R.from_quat(quatpos[:, :4],scalar_first=True)  # convert from (w,x,y,z) to rotation matrix
        mats[:, :3, :3] = r.as_matrix()
        mats[:, 3, 3] = 1.0
        return mats
    
    def _check_xpos(self, model, data, target_name: str):
        eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_name)
        eef_pos = data.xpos[eef_id]
        print(f"{target_name} position: {eef_pos}")

    def _generate_pos(self, t):
        """Generate positions for the arms"""
        pose = np.tile(np.eye(4, dtype=np.float32)[None], (2, 1, 1))
        
        amplitude = 0.2
        center = -0.235
        frequency = 0.1
        x_pos = center + amplitude * np.sin(2 * np.pi * frequency * t)

        # Apply the offsets to the home matrices
        pose = self.home_mat

        return pose

if __name__ == "__main__":
    leg_teleop = LegTeleop()
    leg_teleop.run_teleop()
    # arm_teleop.test_run_mujoco()