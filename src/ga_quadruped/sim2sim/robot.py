import mujoco as mj
import time
import numpy as np

from scipy.spatial.transform import Rotation as R
from ga_quadruped.robot.base_robot import BaseRobot

class SimRobot(BaseRobot):
    def __init__(self, xml_path: str):
        self.spec = mj.MjSpec.from_file(xml_path)

        self.model = self.spec.compile()
        self.data = mj.MjData(self.model)

        default_joint_pos = np.zeros(12)
        init_pos = [0.0, 0.0, 0.1]

        # Default pose and timestep
        self.default_joint_qpos = list(default_joint_pos) if default_joint_pos is not None else [0.0] * (self.model.nq - 7)
        quat = [1, 0, 0, 0]
        self.data.qpos = [init_pos[0], init_pos[1], init_pos[2], *quat, *self.default_joint_qpos]
        self.model.opt.timestep = float(0.005)
        self.model.dof_armature[6:] = [0.01] * len(self.default_joint_qpos)
        mj.mj_forward(self.model, self.data)

        self.kps = self.model.actuator_gainprm[:, 0].copy()

    # ----- BaseRobot API -----
    def step(self, nsteps: int = 1) -> None:
        mj.mj_step(self.model, self.data, nstep=nsteps)

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        self.data.ctrl[:] = np.asarray(ctrl, dtype=float)

    def get_ctrl(self) -> np.ndarray:
        return self.data.ctrl.copy()

    def get_position(self) -> np.ndarray:
        p = self.data.qpos[7:].copy()
        # if self.random_extent["position"]:
            # p += np.random.normal(0.0, self.random_extent["position"], size=p.shape)
        print(self.data.qpos[2])
        return p

    def get_velocity(self) -> np.ndarray:
        v = self.data.qvel[6:].copy()
        # if self.random_extent["velocity"]:
            # v += np.random.normal(0.0, self.random_extent["velocity"], size=v.shape)
        return v

    def get_imu_quat(self) -> np.ndarray:
        # Sensor named "orientation" should provide (w, x, y, z)
        sensor_id = self.model.sensor("orientation").id
        adr = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        return self.data.sensordata[adr: adr + dim].copy()

    def get_gyro(self) -> np.ndarray:
        sensor_id = self.model.sensor("gyro").id
        adr = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        gyro = self.data.sensordata[adr: adr + dim].copy()
        # if self.random_extent["gyro"]:
        #     gyro += np.random.normal(0.0, self.random_extent["gyro"], size=gyro.shape)
        return gyro

    def get_motor_torques(self) -> dict[str, float]:
        t = self.data.actuator_force.copy()
        out = {}
        for i in range(self.model.nu):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            out[f"motor_torque_{name}"] = float(t[i])
        return out
    
    def fall_start(self, idx = None): 
        qpos_recovery = np.load("/home/radon12/Documents/workspace/mjx_rl/qpos_recovery.npy") 
        print("Loaded qpos_recovery.npy with shape:", qpos_recovery.shape) 
        if idx is None: 
            idx = np.random.randint(0, qpos_recovery.shape[0]) 
        init_pos = qpos_recovery[idx].tolist() 
        self.data.qpos = init_pos 
        mj.mj_forward(self.model, self.data)

    def _unit_vector_on_sphere(self, rng):
        v = rng.normal(size=3)
        n = np.linalg.norm(v)
        return v / (n + 1e-9)
    
    def start(self) -> None:
        self.data.ctrl[:] = self.data.qpos[7:]
        self.model.actuator_gainprm[:, 0] = self.kps

    def stop(self) -> None:
        self.model.actuator_gainprm[:, 0] = 0
        mj.mj_forward(self.model, self.data)
            

    def _sample_in_cone_about_x(self, rng, half_angle_rad):
        """
        Returns a unit vector in the robot's LOCAL frame, uniformly distributed
        inside a cone around +X with half-angle = half_angle_rad.
        """
        # Uniform in cone: cos(theta) ~ U[cos(a), 1]
        u1 = rng.random()
        u2 = rng.random()
        ca = np.cos(half_angle_rad)
        cos_theta = 1.0 - u1 * (1.0 - ca)
        sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta**2))
        phi = 2.0 * np.pi * u2
        # Local coordinates: x is the cone axis
        return np.array([cos_theta, sin_theta * np.cos(phi), sin_theta * np.sin(phi)])

    def push(
        self,
        mag=0.5,                       # lower bound of push magnitude
    ):
        """
        Randomizes direction and magnitude of a velocity kick to the root body.
        Scales self.data.qvel[:3] (linear velocity in world frame).

        Tip: If you're using a Gym/Gymnasium-style env, pass self.np_random as rng for reproducibility.
        """
        # Orientation (w, x, y, z) in MuJoCo; scalar_first=True is correct
        quat = self.data.qpos[3:7]
        rot = R.from_quat(quat, scalar_first=True).as_matrix()

        rng = np.random.default_rng()
        
        # Sample direction (world frame)
        direction_world = self._unit_vector_on_sphere(rng)

        # Apply as a velocity kick to the root's linear velocity (world frame)
        self.data.qvel[:3] += direction_world * mag


if __name__ == "__main__":
    from ga_quadruped.robot.quadruped_init import QuadrupedDefaultInitializer

    XML_PATH = "/home/radon12/Documents/ga_quadruped/assets/param/scene.xml" 
    # HOME_POSE = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1, -1.5, -0.1, 1.0, -1.5] 
    theta0 = 0.0 
    theta1 = 0.4 
    theta2 = 1.2 
    # HOME_POSE = [ theta0, -theta1, theta2, -theta0, theta1, -theta2, theta0, -theta1, theta2, -theta0, theta1, -theta2, ] 
    robot = SimRobot(XML_PATH) 
    model = robot.model 
    data = robot.data 
    trunk_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "trunk") 
    
    flag = True

    import mujoco.viewer as mjv 
    import time 
    
    with mjv.launch_passive(model, data) as viewer:
        quadraped_init = QuadrupedDefaultInitializer(
            robot=robot,
            stand_gait=np.array([ theta0, -theta1, theta2, -theta0, theta1, -theta2, theta0, -theta1, theta2, -theta0, theta1, -theta2, ]),
            sit_gait=np.zeros(12),
            viewer=viewer
        )

        quadraped_init.stand()

        for i in range(10000):
            if i % 200 == 0:
                # robot.push()
                if flag:
                    robot.stop()
                else:
                    robot.start()
                flag = not flag

        
            p_body = data.xpos[trunk_id].copy()           # body-frame origin (world)
            p_com  = data.xipos[trunk_id].copy()          # CoM (inertial origin, world)
    
            print(f"trunk origin z = {p_body[2]:.3f} | CoM z = {p_com[2]:.3f} | Δz = {(p_com[2]-p_body[2]):.3f}")
            robot.step()

            viewer.sync()

        time.sleep(1.0)
        quadraped_init.sit() 
        time.sleep(1.0) 