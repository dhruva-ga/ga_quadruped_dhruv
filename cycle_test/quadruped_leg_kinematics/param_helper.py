from ga_can.devices.motor.motor import Motor
from ga_can.transport.socketcan import SocketCANTransport
from ga_can.core.device_manager import DeviceManager
from .kinematics import ParamLegsKinematics
from ga_can.devices.contact_sensor.contact_sensor import ContactSensor
from ga_can.devices.imu.imu import IMU
import numpy as np
import time
from dataclasses import dataclass
from threading import Lock,Thread
from logging import debug
np.set_printoptions(precision=3, suppress=True)

@dataclass
class IMUData:
    rpy: np.ndarray
    accel: np.ndarray
    gyro: np.ndarray

class ParamLegs:

    def __init__(self):
        self.transport = SocketCANTransport("can0")
        self.transport.start()
        self.manager = DeviceManager(self.transport)
        self.kinematics = ParamLegsKinematics(self.manager)
        self.contacter = ContactSensor([0x101,0x102],self.manager)
        self.imu = IMU([0x103,0x104,0x105,0x106],self.manager)
        self.fps = 0
        self.start_time = time.time()
        self.ctrl_lock = Lock()
        self.ctrl = None

    def set_ctrl(self, ctrl):
        with self.ctrl_lock:
            self.ctrl = ctrl.copy()

    def read_contact(self):
        with self.contacter._state_lock:
            return self.contacter._st.contacts.copy()
    
    def read_rpy(self):
        with self.imu._state_lock:
            rpy = self.imu._st.rpy.copy()
        return rpy

    def read_accel(self):
        with self.imu._state_lock:
            accel = self.imu._st.accel.copy()
        return accel
    def read_gyro(self):
        with self.imu._state_lock:
            gyro = self.imu._st.gyro.copy()
        return gyro

    def get_imu_data(self):
        with self.imu._state_lock:
            return IMUData(
                rpy=self.imu._st.rpy.copy(),
                accel=self.imu._st.accel.copy(),
                gyro=self.imu._st.gyro.copy()
            )
        
    def get_kinematics_data(self):
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
                # else:
                #     self.kinematics.query_everything()
            t2 = time.time()
            #print(f"Loop time: {t2 - t1:.4f} seconds")
            if t2 - t1 < 0.01:
                time.sleep(0.01 - (t2 - t1))


if __name__ == "__main__":
    ga_one = ParamLegs()
    time.sleep(2)
    ga_one.start()
    while True:
        time.sleep(1)




    
