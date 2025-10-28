import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

import yaml
from typing import List, Literal, Dict

def load_deploy_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)    # or yaml.UnsafeLoader

def reorder_by_joint_ids_map(
    src: List[float],
    joint_ids_map: List[int],
    map_kind: Literal["model_to_action","action_to_model"]="model_to_action",
) -> List[float]:
    n = len(joint_ids_map)
    if len(src) != n:
        raise ValueError(f"Length mismatch: len(src)={len(src)} vs len(joint_ids_map)={n}")
    out = [0.0]*n
    if map_kind == "model_to_action":
        for model_idx, action_idx in enumerate(joint_ids_map):
            out[model_idx] = float(src[action_idx])
    else:
        for action_idx, model_idx in enumerate(joint_ids_map):
            out[model_idx] = float(src[action_idx])
    return out

def get_joint_names_in_model_order(model: mj.MjModel) -> List[str]:
    """1-DoF joints in XML/model order (works for hinge/slide)."""
    names = []
    for j in range(model.njnt):
        names.append(mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j))
    return names

def map_joint_to_position_actuator(model: mj.MjModel) -> dict[int, int]:
    """
    Returns {joint_id -> actuator_id} for actuators transmitting to that joint
    (for position servos).
    """
    j2a = {}
    for a in range(model.nu):
        print(model.actuator_trntype.shape)
        trn0 = int(model.actuator_trnid[a, 0])       # first target of actuator a
        trntype0 = int(model.actuator_trntype[a]) # type of that target
        if trn0 >= 0 and trntype0 == mj.mjtTrn.mjTRN_JOINT:
            j2a[trn0] = a
    return j2a

def set_actuator_kp(model, actuator_id: int, kp: float):
    # gainprm has shape (nu, 10)
    model.actuator_gainprm[actuator_id, 0] = float(kp)

def set_joint_damping(model: mj.MjModel, joint_id: int, damping: float):
    # Each joint can have 1+ DoFs; hinge/slide have 1
    dof = model.jnt_dofadr[joint_id]
    model.dof_damping[dof] = float(damping)

def apply_pd_from_yaml(
    model: mj.MjModel,
    yaml_path: str,
    map_kind: Literal["model_to_action","action_to_model"]="model_to_action",
):
    cfg = load_deploy_yaml(yaml_path)
    # joint_ids_map = list(cfg["joint_ids_map"])
    joint_ids_map = list(range(12))
    stiffness = list(cfg["stiffness"])
    damping   = list(cfg["damping"])

    # Align gains to *model joint order*
    kp_model = reorder_by_joint_ids_map(stiffness, joint_ids_map, map_kind)
    kd_model = reorder_by_joint_ids_map(damping,   joint_ids_map, map_kind)

    # Names (optional; useful for sanity/logging)
    model_joint_names = get_joint_names_in_model_order(model)

    # Find which actuator drives which joint
    j2a = map_joint_to_position_actuator(model)

    # Apply
    for j_id, (kp, kd) in enumerate(zip(kp_model, kd_model)):
        # Set joint physical damping (KD)
        set_joint_damping(model, j_id, kd)

        # If there is a position actuator for this joint, set its kp
        a_id = j2a.get(j_id, None)
        if a_id is not None:
            set_actuator_kp(model, a_id, kp)
        else:
            # No actuator attached (or non-joint transmission) — skip
            pass

    # Recompute derived constants after structural edits
    data = mj.MjData(model)
    mj.mj_setConst(model, data)

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
        self.spec = mj.MjSpec.from_file(xml_path)
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
        # apply_pd_from_yaml(self.model, "/home/radon12/Documents/ga_quadruped/src/ga_quadruped/policy/deploy.yaml", map_kind="model_to_action")
        self.data = mj.MjData(self.model)
        
        model = self.model
        joint_info = []
        for i in range(model.njnt):
            joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
            qpos_start_idx = model.jnt_qposadr[i]
            
            # Calculate number of qpos entries for this joint based on joint type
            joint_type = model.jnt_type[i]
            if joint_type == mj.mjtJoint.mjJNT_FREE:
                qpos_size = 7  # 3 position + 4 quaternion
            elif joint_type == mj.mjtJoint.mjJNT_BALL:
                qpos_size = 4  # quaternion
            elif joint_type in [mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE]:
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
            actuator_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            joint_id = self.model.actuator_trnid[i, 0]  # Joint ID this actuator controls
            joint_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, joint_id)
            
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
        
     
        HOME_POSE = default_joint_pos.copy()
        
        self._stand_gait = np.array(HOME_POSE)
        self._sit_gait = np.zeros(12)

        # exit()

        # self.joint_ids = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]

        self.default_joint_qpos = default_joint_pos
        rotation = R.from_euler('x', 0, degrees=True)
        quat = rotation.as_quat(scalar_first=True)
        quat = [1,0,0,0]
        self.data.qpos = [init_pos[0], init_pos[1], init_pos[2], quat[0], quat[1], quat[2], quat[3]] + self.default_joint_qpos
        self.model.opt.timestep = float(0.005)
        self.model.dof_armature[6:] = [0.01] * len(self.default_joint_qpos)
        mj.mj_forward(self.model, self.data)
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
        mj.mj_step(self.model, self.data, nstep=nsteps)

    def set_ctrl(self, ctrl: np.ndarray):
        self.data.ctrl[:] = ctrl.copy()
        # self.data.ctrl[:] = ctrl.copy()[self.joint_ids]

    def get_ctrl(self) -> np.ndarray:
        return self.data.ctrl.copy()
        
    # def get_torque(self) -> np.ndarray:
    #     """Get joint torques (excluding root)."""
    #     t = self.data.qfrc_actuator[6:].copy()
    #     return t

    def get_position(self) -> np.ndarray:
        """Get joint positions (excluding root)."""
        p = self.data.qpos[7:].copy()
        p += np.random.normal(0.0, self.random_extent["position"], size=p.shape)  # Add noise
        # p = p[self.joint_ids]
        trunk_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "trunk")
        p_body = self.data.xpos[trunk_id] 
        p_com = self.data.xipos[trunk_id]
        print(f"trunk origin z = {p_body[2]:.3f} | CoM z = {p_com[2]:.3f} | Δz = {(p_com[2]-p_body[2]):.3f}")

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
    
    def get_motor_torques(self) -> np.ndarray:
        """Get motor torques (excluding root)."""
        t = self.data.actuator_force.copy()
        out = {}
        for i in range(self.model.nu):  # nu = number of actuators
            actuator_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            out["motor_toque_" + actuator_name] = float(t[i])
        return out
    
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

def _stand(robot, viewer):
    jnt_pos = robot.get_ctrl()
    for i in range(200):
        rate = min(i/200, 1)
        des_gait = jnt_pos * (1 - rate) + robot._stand_gait * rate
        robot.set_ctrl(des_gait)
        robot.step(nsteps=4)
        viewer.sync()
        time.sleep(0.01)

def _sit(robot, viewer):
    jnt_pos = robot.get_ctrl()
    for i in range(200):
        rate = min(i/200, 1)
        des_gait = jnt_pos * (1 - rate) + robot._sit_gait * rate
        robot.set_ctrl(des_gait)
        robot.step(nsteps=4)
        viewer.sync()
        time.sleep(0.01)


import numpy as np
import mujoco as mjfeet_

def _as_f64(x, n=None):
    a = np.asarray(x, dtype=np.float64)
    return a if n is None else a.reshape(n)

def _as_mat9_f64(R3x3):
    M = np.asarray(R3x3, dtype=np.float64)
    return M.reshape(9) if M.shape == (3, 3) else M.astype(np.float64).reshape(9)

def _as_rgba_f32(rgba):
    a = np.asarray(rgba, dtype=np.float32)
    return a.reshape(4)

def _make_geom(user_scn, gtype, pos, size, mat=None, rgba=(1,1,1,1)) -> mj.MjvGeom:
    # print(user_scn.geoms[0])
    # print(user_scn.ngeom)
    # exit()

    # print(user_scn.geoms[user_scn.ngeom])
    # exit()
    # print(user_scn.ngeom)
    user_scn.ngeom += 1
    mj.mjv_initGeom(
        user_scn.geoms[user_scn.ngeom],  # geom to initialize
        gtype,
        _as_f64(size, 3),        # float64 [3]
        _as_f64(pos, 3),         # float64 [3]
        _as_mat9_f64(mat),       # float64 [9]
        _as_rgba_f32(rgba),      # float32 [4]
    )


def add_marker_sphere(user_scn, pos, size_xyz, rgba):
    g = _make_geom(user_scn, mj.mjtGeom.mjGEOM_SPHERE, pos, size_xyz, np.eye(3).flatten(), rgba)
    # _append_geom(user_scn, g)

def _rot_from_dir(dir_vec):
    z = np.asarray(dir_vec, dtype=np.float64)
    z /= (np.linalg.norm(z) + 1e-9)
    tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(tmp, z)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x = np.cross(tmp, z); x /= (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)

def add_marker_cylinder(user_scn, p0, p1, radius, rgba):
    p0 = np.asarray(p0, dtype=np.float64); p1 = np.asarray(p1, dtype=np.float64)
    vec  = p1 - p0
    L    = np.linalg.norm(vec) + 1e-9
    half = 0.5 * L
    mid  = 0.5 * (p0 + p1)
    R    = _rot_from_dir(vec)
    # cylinder size = [radius, radius, half_length]
    g = _make_geom(user_scn, mj.mjtGeom.mjGEOM_CYLINDER, mid, [radius, radius, half], R, rgba)

def add_axis_arrows(user_scn, origin, R_body, scale=0.06):
    radius = 0.003
    half   = scale
    def _arrow(dir_vec, rgba):
        R = _rot_from_dir(dir_vec)
        g = _make_geom(user_scn, mj.mjtGeom.mjGEOM_ARROW, origin, [radius, radius, half], R, rgba)
    _arrow(R_body[:, 0] * scale, [1, 0, 0, 1])  # X
    _arrow(R_body[:, 1] * scale, [0, 1, 0, 1])  # Y
    _arrow(R_body[:, 2] * scale, [0, 0, 1, 1])  # Z



if __name__ == "__main__":
    XML_PATH = "/home/radon12/Documents/ga_quadruped/assets/param/scene.xml"
    
    # HOME_POSE = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1, -1.5, -0.1, 1.0, -1.5]
    theta0 = 0.0
    theta1 = 0.4
    theta2 = 1.2
    
    HOME_POSE = [           
        theta0,
        -theta1,
        theta2,
        -theta0,
        theta1,
        -theta2,
        theta0,
        -theta1,
        theta2,
        -theta0,
        theta1,
        -theta2,
    ]

    robot = Robot(XML_PATH, randomisation=False,
                  default_joint_pos=HOME_POSE,
                  init_pos=[0, 0, 0.5])
    
    model = robot.model
    data = robot.data
    trunk_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "trunk")

    import mujoco.viewer as mjv

    import time

    # with mjv.launch_passive(model, data) as viewer:
    #     _sit(robot, viewer)
    #     time.sleep(1.0)
    #     _stand(robot, viewer)
    #     time.sleep(1.0)

    # exit()

    np.set_printoptions(precision=3, suppress=True)
    # Use MuJoCo's official viewer
    with mjv.launch_passive(model, data) as viewer:
        # Optionally tweak UI or camera:
        # viewer.cam.distance = 2.5
        # viewer.cam.azimuth = 90
        # viewer.cam.elevation = -15

        while viewer.is_running():
            # drive to your default pose if desired
            robot.set_ctrl(robot.default_joint_qpos)
            mj.mj_step(model, data)

            print(robot.get_motor_torques())

            # Clear user scene each frame before adding new geoms
            viewer.user_scn.ngeom = 0

            # world poses
            p_body = data.xpos[trunk_id].copy()           # body-frame origin (world)
            # p_body[2] -= 0.07                          # offset above trunk for visibility2
            p_com  = data.xipos[trunk_id].copy()          # CoM (inertial origin, world)
            R_body = data.xmat[trunk_id].reshape(3, 3).copy()

            # markers: origin (blue) and CoM (red)
            add_marker_sphere(viewer.user_scn, p_body, [0.02, 0.02, 0.02], [0.1, 0.6, 1.0, 1.0])
            add_marker_sphere(viewer.user_scn, p_com,  [0.025, 0.025, 0.025], [1.0, 0.3, 0.3, 1.0])

            # line (thin cylinder) between them
            add_marker_cylinder(viewer.user_scn, p_body, p_com, radius=0.004, rgba=[0.9, 0.85, 0.2, 0.8])

            # local axes at the body origin
            add_axis_arrows(viewer.user_scn, p_body, R_body, scale=0.06)

            # HUD printout to console (optional)
            print(f"trunk origin z = {p_body[2]:.3f} | CoM z = {p_com[2]:.3f} | Δz = {(p_com[2]-p_body[2]):.3f}")

            # Render one frame
            viewer.sync()
            # time.sleep(0.1)

    # with launch_passive(robot.model, robot.data) as viewer:
    #     # Sweeps every hinge/slide joint, printing: joint index, qpos index, limits, and actuators.
    #     robot.sweep_hinge_joints(viewer=viewer, seconds_per_joint=1.0, steps=120, pause=0.01) 