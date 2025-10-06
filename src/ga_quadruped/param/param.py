from ga_can.devices.motor.motor import Motor
from ga_can.transport.socketcan import SocketCANTransport
from ga_can.core.device_manager import DeviceManager
from ga_can.devices.contact_sensor.contact_sensor import ContactSensor
from ga_can.devices.imu.imu import IMU
import numpy as np
import time
from dataclasses import dataclass
from threading import Lock,Thread
from logging import debug

from ga_quadruped.param.kinematics import ParamLegsKinematics
np.set_printoptions(precision=3, suppress=True)

CAN_ICM_GYRO = 0x102
CAN_MPU_GYRO = 0x104
CAN_BNO_GYRO = 0x107
CAN_BNO_QUAT = 0x113

@dataclass
class IMUData:
    rpy: np.ndarray
    gyro: np.ndarray
    quat: np.ndarray


class Param:

    def __init__(self):
        self.transport = SocketCANTransport("can0")
        self.transport.start()
        self.manager = DeviceManager(self.transport)
        self.kinematics = ParamLegsKinematics(self.manager)
        # self.contacter = ContactSensor([0x101,0x102],self.manager)
        self.transport2 = SocketCANTransport("can1")
        self.transport2.start()
        self.manager2 = DeviceManager(self.transport2)
        self.imu = IMU([CAN_BNO_GYRO, CAN_BNO_QUAT],self.manager2)
        self.fps = 0
        self.start_time = time.time()
        self.ctrl_lock = Lock()
        self.ctrl = None

        theta = 0.4
        theta2 = 1.2
        HOME_POSE = np.array([
            0.0,   
            -theta,   
            theta2, 
            0.0,
            theta,
            -theta2,
            0.0,
            -theta,
            theta2,
            0.0,
            theta,
            -theta2
        ])
        self._stand_gait = HOME_POSE.copy()
        self._sit_gait = np.zeros(12)

    def set_ctrl(self, ctrl):
        with self.ctrl_lock:
            self.ctrl = ctrl.copy()
    
    def get_ctrl(self):
        with self.ctrl_lock:
            if self.ctrl is not None:
                return self.ctrl.copy()
            else:
                return None

    # def read_contact(self):
    #     with self.contacter._state_lock:
    #         return self.contacter._st.contacts.copy()
    
    # def read_rpy(self):
    #     with self.imu._state_lock:
    #         rpy = self.imu._st.rpy.copy()
    #     return rpy

    # def read_accel(self):
    #     with self.imu._state_lock:
    #         accel = self.imu._st.accel.copy()
        # return accel
    
    # def read_gyro(self):param_imu
    #     with self.imu._state_lock:
    #         gyro = self.imu._st.gyro.copy()
    #     return gyro
    
    # def read_quat(self):
    #     with self.imu._state_lock:
    #         quat = self.imu._st.quat.copy()
    #     return quat

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

    def start(self):
        self.thread = Thread(target=self.run, daemon=True)
        self.thread.start()
        print("GaOne Interface started")
    
    def run(self):
        while True:
            t1 = time.time()
            with self.ctrl_lock:
                if self.ctrl is not None:
                    self.kinematics.set_angles(self.ctrl)
                else:
                    self.kinematics.query_everything()
            t2 = time.time()
            #print(f"Loop time: {t2 - t1:.4f} seconds")
            if t2 - t1 < 0.01:
                time.sleep(0.01 - (t2 - t1))

    def _stand(self):
        jnt_pos = self.ctrl.copy() if self.ctrl is not None else np.array(self.kinematics.read_data().angles)
        for i in range(200):
            rate = min(i/200, 1)
            des_gait = jnt_pos * (1 - rate) + self._stand_gait * rate
            self.ctrl = des_gait
            time.sleep(0.01)

    def _sit(self):
        jnt_pos = self.ctrl.copy() if self.ctrl is not None else np.array(self.kinematics.read_data().angles)
        for i in range(200):
            rate = min(i/200, 1)
            des_gait = jnt_pos * (1 - rate) + self._sit_gait * rate
            self.ctrl = des_gait
            time.sleep(0.01)

# def compute_gravity_orientation(quat: np.ndarray) -> np.ndarray:
#     gx = 2 * (quat[1] * quat[3] - quat[0] * quat[2])
#     gy = 2 * (quat[0] * quat[1] + quat[2] * quat[3])
#     gz = quat[0] * quat[0] - quat[1] * quat[1] - quat[2] * quat[2] + quat[3] * quat[3]
#     return np.array([-gx, -gy, -gz], dtype=np.float32)


if __name__ == "__main__":
    ga_one = Param()
    time.sleep(2)
    ga_one.start()
    time.sleep(2)
    ga_one._sit()
    time.sleep(2)
    ga_one._stand()

    np.set_printoptions(precision=3, suppress=True)
    t = 10
    total = 50 * t
    for i in range(total):
        print(ga_one.get_imu_data().gyro)
        time.sleep(0.02)  # Adjust the sleep time as needed

    ga_one._sit()
    

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




    
