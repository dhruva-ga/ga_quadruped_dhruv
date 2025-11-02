from dataclasses import dataclass
from threading import Lock, Thread
from ga_quadruped.param.kinematics import ParamLegsKinematics
from ga_can.transport.socketcan import SocketCANTransport
from ga_can.core.device_manager import DeviceManager
from ga_can.devices.imu.imu import IMU

import numpy as np

from ga_quadruped.robot.base_robot import BaseRobot
from ga_quadruped.robot.quadruped_init import QuadrupedDefaultInitializer
import time


CAN_ICM_GYRO = 0x102
CAN_MPU_GYRO = 0x104
CAN_BNO_GYRO = 0x107
CAN_BNO_QUAT = 0x113

np.set_printoptions(precision=3, suppress=True)
@dataclass
class IMUData:
    rpy: np.ndarray
    gyro: np.ndarray
    quat: np.ndarray



class Param(BaseRobot):
    """Real robot implementation with a uniform interface.

    Retains the name `Param` for back-compat but now conforms to `BaseRobot`.
    """

    def __init__(self):
        self.transport = SocketCANTransport("can0"); self.transport.start()
        self.manager = DeviceManager(self.transport)
        self.kinematics = ParamLegsKinematics(self.manager)

        self.transport2 = SocketCANTransport("can1"); self.transport2.start()
        self.manager2 = DeviceManager(self.transport2)
        # BNO gyro + quat frames
        self.imu = IMU([CAN_BNO_GYRO, CAN_BNO_QUAT], self.manager2)

        self._ctrl_lock = Lock()
        self._ctrl = np.zeros(12, dtype=float)

        self._thread: Thread | None = None

    # ----- BaseRobot API -----
    def set_ctrl(self, ctrl: np.ndarray) -> None:
        with self._ctrl_lock:
            self._ctrl = np.asarray(ctrl, dtype=float)

    def get_ctrl(self) -> np.ndarray | None:
        with self._ctrl_lock:
            return None if self._ctrl is None else self._ctrl.copy()

    def get_position(self) -> np.ndarray:
        kd = self.kinematics.read_data()
        return np.asarray(kd.angles, dtype=float)

    def get_velocity(self) -> np.ndarray:
        kd = self.kinematics.read_data()
        return np.asarray(kd.velocity, dtype=float)

    def get_imu_quat(self) -> np.ndarray:
        with self.imu._state_lock:
            return self.imu._st.quat.copy()

    def get_gyro(self) -> np.ndarray:
        with self.imu._state_lock:
            return (self.imu._st.bno_gyro.copy()) * np.pi / 180.0

    def get_motor_torques(self) -> dict[str, float]:
        kd = self.kinematics.read_data()
        return {f"motor_torque_{n}": float(t) for n, t in zip(kd.motor_name, kd.torque)}

    def get_motor_temps(self) -> dict[str, float]:
        kd = self.kinematics.read_data()
        return {f"motor_temp_{n}": float(t) for n, t in zip(kd.motor_name, kd.motor_temp)}
    
    # Backwards compatibility methods
    def get_imu_data(self):
        with self.imu._state_lock:
            # print(self.imu._st.rpy, self.imu._st.quat, self.imu._st.bno_gyro)
            return IMUData(
                rpy=self.imu._st.rpy.copy() * np.pi / 180,
                quat=self.imu._st.quat.copy(),
                gyro=self.imu._st.bno_gyro.copy() * np.pi / 180
            )
        
    def get_kinematics_data(self):
        print("Getting kinematics data")
        return self.kinematics.read_data()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()
        print("GA Param interface started")

    def step(self, nsteps: int = 1) -> None:
        pass

    # ----- internals -----
    def _run(self) -> None:
        while True:
            t1 = time.time()
            with self._ctrl_lock:
                if self._ctrl is not None:
                    self.kinematics.set_angles(self._ctrl)
                else:
                    self.kinematics.query_everything()
            t2 = time.time()
            if t2 - t1 < 0.01:
                time.sleep(0.01 - (t2 - t1))



if __name__ == "__main__":
    import argparse
    import time

    args = argparse.ArgumentParser()
    args.add_argument('--zero', action='store_true', help='Start with zero joint positions')
    args.add_argument('--read', action='store_true', help='Continuously read and print kinematics data')

    # args.add_argument('--stand', action='store_true', help='Stand')
    args = args.parse_args()

        
    theta0 = 0.0
    # theta1 = 0.45
    theta1 = 0.4
    theta2 = 1.2
    # theta2 = 1.4
    HOME_POSE = [theta0, -theta1, theta2, -theta0, theta1, -theta2, theta0, -theta1, theta2, -theta0, theta1, -theta2]
    
    robot = Param()

    quadraped_init = QuadrupedDefaultInitializer(
        robot=robot,
        stand_gait=np.array(HOME_POSE),
        sit_gait=np.zeros(12),
    )

    
    robot.start()
    time.sleep(0.1)

    if args.read:
        print(robot.get_kinematics_data().angles)
        exit()
    else:
        quadraped_init.sit()
        time.sleep(1.0)
        if not args.zero:
            quadraped_init.stand()

    np.set_printoptions(precision=3, suppress=True)
    t = 1000
    total = 50 * t
    try:
        for i in range(total):
            print(robot.get_kinematics_data())
            time.sleep(0.02)  # Adjust the sleep time as needed
    except KeyboardInterrupt:
        quadraped_init.sit()

    time.sleep(2)
    quadraped_init.sit()
    

    # exit()

    # theta = 0.4
    # theta2 = 1.5
    # HOME_POSE = np.array([
    #     0.0,   
    #     -theta,   
    #     theta2, 
    #     0.0,
    #     theta,
    #     -theta2,
    #     0.0,
    #     -theta,
    #     theta2,
    #     0.0,
    #     theta,
    #     -theta2
    # ])

    # current_pose = ga_one.get_kinematics_data().angles
    # diff = HOME_POSE - current_pose
    # t = 1
    # total = int(50 * t)
    # for i in range(total): 
    #     ctrl = diff*(i/total) + current_pose 
    #     ga_one.set_ctrl(ctrl)
    #     time.sleep(0.02)  # Adjust the sleep time as needed


    # time.sleep(3)
    # current_pose = ctrl.copy()
    # #current_pose = ga_one.get_kinematics_data().angles
    # SIT_POSE = np.zeros(12)
    # diff = SIT_POSE - current_pose
    # t = 10
    # total = 50 * t
    # for i in range(total):
    #     ctrl = diff*(i/total) + current_pose 
    #     ga_one.set_ctrl(ctrl)
    #     time.sleep(0.02)  # Adjust the sleep time as needed


    # while True:
    #     time.sleep(1)
    #     print(ga_one.get_kinematics_data())




    
