import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

class Robot:
    """
    Initialize the Robot instance by loading the Mujoco model and data, and optionally applying randomization.
    Args:
        xml_path (str): Path to the Mujoco XML model file.
        randomisation (bool, optional): If True, randomizes actuator gains, friction, and sensor noise parameters.
        randomisation_params (dict, optional): Dictionary specifying the extent of randomization for various parameters.
            The expected structure is:
                {
                    "gains": float,          # Percentage of actuator gain randomization
                    "friction": float,       # Percentage of floor friction randomization
                    "position": float,       # STD of position sensor noise
                    "velocity": float,       # STD of velocity sensor noise
                    "accelerometer": float,  # STD of accelerometer sensor noise
                    "gyro": float,           # STD of gyroscope sensor noise
                    "imu_rpy": float         # STD of IMU roll/pitch/yaw noise (in radians)
            If not provided, default values are used.
    """

    def __init__(self, xml_path: str, randomisation: bool = False, randomisation_params: dict = None, default_joint_pos = None, init_pos = [0.0, 0.0, 0.0]):
        # Load model and data
        self.random_extent = {
            "gains": 0.0,
            "friction": 0.0,
            "position": 0.00,
            "velocity": 0.00,
            "accelerometer": 0.00,
            "gyro": 0.00,
            "imu_rpy": 0.00,
        }
        self.spec = mujoco.MjSpec.from_file(xml_path)
        if randomisation:
            self.random_extent = randomisation_params if randomisation_params else {
            "gains": 0.1,
            "friction": 0.1,
            "position": 0.03,
            "velocity": 1.50,
            "accelerometer": 0.20,
            "gyro": 0.20,
            "imu_rpy": 0.10}
            self.randomize()
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        
        model = self.model
        joint_info = []
        for i in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            qpos_start_idx = model.jnt_qposadr[i]
            
            # Calculate number of qpos entries for this joint based on joint type
            joint_type = model.jnt_type[i]
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                qpos_size = 7  # 3 position + 4 quaternion
            elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                qpos_size = 4  # quaternion
            elif joint_type in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                qpos_size = 1  # single value
            else:
                qpos_size = 1  # default
            
            joint_info.append({
                'name': joint_name,
                'joint_id': i,
                'qpos_start': qpos_start_idx,
                'qpos_size': qpos_size,
                'qpos_range': f"{qpos_start_idx}:{qpos_start_idx + qpos_size}"
            })

        actuator_info = []
    
        for i in range(self.model.nu):  # nu = number of actuators
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            joint_id = self.model.actuator_trnid[i, 0]  # Joint ID this actuator controls
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            
            actuator_info.append({
                'actuator_id': i,
                'actuator_name': actuator_name,
                'joint_id': joint_id,
                'joint_name': joint_name,
                'ctrl_index': i  # Index in ctrl array
            })

        # Print the information
        for joint in joint_info:
            print(f"Joint '{joint['name']}': qpos[{joint['qpos_range']}]")

        for info in actuator_info:
            print(f"ctrl[{info['ctrl_index']}] -> Actuator '{info['actuator_name']}' -> Joint '{info['joint_name']}'")
        
        # exit()

        # self.joint_ids = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]

        self.default_joint_qpos = default_joint_pos
        rotation = R.from_euler('y', 2, degrees=True)
        quat = rotation.as_quat(scalar_first=True)
        quat = [1,0,0,0]
        self.data.qpos = [init_pos[0], init_pos[1], init_pos[2], quat[0], quat[1], quat[2], quat[3]] + self.default_joint_qpos
        self.model.opt.timestep = float(0.005)
        self.model.dof_armature[6:] = [0.01] * len(self.default_joint_qpos)
        mujoco.mj_forward(self.model, self.data)
        for actuator in self.spec.actuators:
            print(type(actuator))
            print(actuator.name, actuator.gaintype, actuator.gainprm, actuator.biasprm)

        # exit()

    def randomize(self):
        """Randomize the robot's actuator params and floor friction."""
        for actuator in self.spec.actuators:
            curr_kp = np.abs(actuator.gainprm[0])
            curr_kd = np.abs(actuator.biasprm[2])
            kp_rand = np.random.uniform(-self.random_extent["gains"] * curr_kp, self.random_extent["gains"] * curr_kp)
            kd_rand = np.random.uniform(-self.random_extent["gains"] * curr_kd, self.random_extent["gains"] * curr_kd)
            actuator.gainprm[0] = actuator.gainprm[0] + kp_rand
            actuator.biasprm[1] = actuator.biasprm[1] - kp_rand
            actuator.biasprm[2] = actuator.biasprm[2] - kd_rand
        floor_sliding_friction = self.spec.worldbody.bodies[0].geoms[0].friction[0]
        delta_friction = np.random.uniform(-self.random_extent["friction"] * floor_sliding_friction, self.random_extent["friction"] * floor_sliding_friction)
        self.spec.worldbody.bodies[0].geoms[0].friction = self.spec.worldbody.bodies[0].geoms[0].friction + delta_friction
        
    # def forward(self,qpos):
    #     self.data.qpos = qpos.copy()
    #     mujoco.mj_forward(self.model, self.data)

    def step(self, nsteps: int = 1):
        """Advance the simulation by nsteps."""
        mujoco.mj_step(self.model, self.data, nstep=nsteps)

    def set_ctrl(self, ctrl: np.ndarray):
        self.data.ctrl[:] = ctrl.copy()
        # self.data.ctrl[:] = ctrl.copy()[self.joint_ids]

    # def get_torque(self) -> np.ndarray:
    #     """Get joint torques (excluding root)."""
    #     t = self.data.qfrc_actuator[6:].copy()
    #     return t

    def get_position(self) -> np.ndarray:
        """Get joint positions (excluding root)."""
        p = self.data.qpos[7:].copy()
        p += np.random.normal(0.0, self.random_extent["position"], size=p.shape)  # Add noise
        # p = p[self.joint_ids]
        return p
    
    def get_velocity(self) -> np.ndarray:
        """Get joint velocities (excluding root)."""
        v = self.data.qvel[6:].copy()
        v += np.random.normal(0.0, self.random_extent["velocity"], size=v.shape)  # Add noise
        # v = v[self.joint_ids]
        return v
    
    def get_lin_vel(self):
        return self.data.qvel[0:3].copy()
    
    def get_ang_vel(self):
        return self.data.qvel[3:6].copy()
    
    def _get_sensor_data(self,name):
        """Get sensor data by name."""
        sensor_id = self.model.sensor(name).id
        sensor_adr = self.model.sensor_adr[sensor_id]
        sensor_dim = self.model.sensor_dim[sensor_id]
        return self.data.sensordata[sensor_adr : sensor_adr + sensor_dim].copy()
    
    def get_accelerometer(self) -> np.ndarray:
        """Get accelerometer data."""
        d1 = self._get_sensor_data("accelerometer")
        d1 += np.random.normal(0.0, self.random_extent["accelerometer"], size=d1.shape)
        # d1[:3] = 0
        return d1
    
    def get_gyro(self) -> np.ndarray:
        """Get gyroscope data."""
        gyro = self._get_sensor_data("gyro")
        gyro += np.random.normal(0.0, self.random_extent["gyro"], size=gyro.shape)
        return gyro.copy()
    
    def get_imu_quat(self) -> np.ndarray:
        """Get IMU roll, pitch, yaw in radians."""
        imu_orientation = self._get_sensor_data("orientation")
        # rot = R.from_quat(imu_orientation,scalar_first=True)
        # rpy = rot.as_euler('xyz', degrees=False)
        # noise = np.random.normal(0.0, self.random_extent["imu_rpy"], size=(3,))
        # print("RPY Noise ", noise)
        # range = 5
        # noise = np.clip(noise, -range*(np.pi/180), range*(np.pi/180.0))
        # rpy = rpy + noise
        return imu_orientation

    def get_gravity_vector(self) -> np.ndarray:
        """Get the gravity vector. Actually up vector"""
        return self._get_sensor_data("gravity_vector").copy()

    
    # def get_contacts(self) -> np.ndarray:
    #     """Get collision state between each foot and the floor."""
    #     contacts = []
    #     for geom_id in self.foot_geom_ids:
    #         colliding = False
    #         for i in range(self.data.ncon):
    #             c = self.data.contact[i]
    #             if (c.geom1 == self.floor_geom_id and c.geom2 == geom_id) or (
    #                 c.geom2 == self.floor_geom_id and c.geom1 == geom_id
    #             ):
    #                 colliding = True
    #                 break
    #         contacts.append(float(colliding))
    #     return np.array(contacts, dtype=np.float32)

    def _joint_qpos_slice(self, jnt_id: int):
        """Return (start, size) into qpos for a joint."""
        jt = self.model.jnt_type[jnt_id]
        start = self.model.jnt_qposadr[jnt_id]
        if jt == mujoco.mjtJoint.mjJNT_FREE:   size = 7
        elif jt == mujoco.mjtJoint.mjJNT_BALL: size = 4
        else:                                  size = 1  # hinge/slide
        return start, size, jt

    def _joint_limits(self, jnt_id: int):
        """Return (lo, hi) limits for hinge/slide joints; small safe span if unlimited."""
        jt = self.model.jnt_type[jnt_id]
        if jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            if int(self.model.jnt_limited[jnt_id]) == 1:
                lo, hi = self.model.jnt_range[jnt_id]  # shape (2,)
                lo, hi = float(lo), float(hi)
                # guard against bogus/degenerate ranges
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    return lo, hi
            # unlimited or bad range → use a small safe sweep
            return (-0.5, 0.5) if jt == mujoco.mjtJoint.mjJNT_HINGE else (-0.05, 0.05)
        # ball/free joints aren't swept here
        return 0.0, 0.0

    def sweep_hinge_joints(self, viewer=None, seconds_per_joint=1.0, steps=100, pause=0.01):
        """
        For each hinge/slide joint:
          - sweep its qpos across limits without stepping physics
          - print mapping: joint index → qpos slice → actuator(s)
        """
        hz = max(1, steps)
        for j in range(self.model.njnt):
            start, size, jt = self._joint_qpos_slice(j)
            if size != 1:   # skip free/ball for clarity
                continue

            lo, hi = self._joint_limits(j)
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)

            # find actuator(s) driving this joint, if any
            driven_by = []
            for a in range(self.model.nu):
                if self.model.actuator_trnid[a, 0] == j:
                    driven_by.append(a)
            print(f"[{j:02d}] Joint '{name}': qpos[{start}] in [{lo:.3f}, {hi:.3f}]  actuators={driven_by}")

            # keep root pose & other joints; sweep only this joint
            base_qpos = self.data.qpos.copy()
            path = np.linspace(lo, hi, hz//2, endpoint=True)
            path = np.concatenate([path, path[::-1]])  # there & back

            # duration pacing
            import time
            tick = (seconds_per_joint / max(1, len(path)))

            for val in path:
                self.data.qpos[:] = base_qpos
                self.data.qpos[start] = val
                mujoco.mj_forward(self.model, self.data)  # kinematic update only
                if viewer is not None:
                    viewer.sync()
                    time.sleep(max(0.0, tick))

if __name__ == "__main__":
    XML_PATH = "/home/radon12/Documents/go1_temp/src/go1/assets/param/scene.xml"
    theta = 0.4
    theta2 = 1.2
    HOME_POSE = [-0.0, -0.3, theta2, 0.0, 0.3, -theta2, 0.0, -theta, theta2, -0.0, theta, -theta2]
    robot = Robot(XML_PATH, randomisation=False,
                  default_joint_pos=HOME_POSE,
                  init_pos=[0, 0, 0.4])
    
    # XML_PATH = "/home/radon12/Documents/go1_temp/src/go1/assets//go1/scene.xml"
    # robot = Robot(XML_PATH, randomisation=False,
    #               default_joint_pos=[0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1, -1.5, -0.1, 1.0, -1.5],
    #               init_pos=[0, 0, 0.45])
    
    from mujoco.viewer import launch_passive
    
    model = robot.model
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    data = robot.data

    np.set_printoptions(precision=3, suppress=True)
    import time
    with launch_passive(robot.model, robot.data) as viewer:
        while True:
            robot.set_ctrl(robot.default_joint_qpos)
            robot.step(4)
            viewer.sync()
            time.sleep(0.02)
            print(data.xpos[body_id], data.xquat[body_id])
            # time.sleep(1)

    # with launch_passive(robot.model, robot.data) as viewer:
    #     # Sweeps every hinge/slide joint, printing: joint index, qpos index, limits, and actuators.
    #     robot.sweep_hinge_joints(viewer=viewer, seconds_per_joint=1.0, steps=120, pause=0.01) 