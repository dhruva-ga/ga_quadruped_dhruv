from ga_can.devices.motor.motor import Motor,MotorType
import numpy as np
from dataclasses import dataclass

@dataclass
class KinematicsData:
    angles: np.ndarray
    velocity: np.ndarray
    torque: np.ndarray
    motor_times: np.ndarray  # Added to store motor times

class GaOneArmsKinematics:
    def __init__(self,manager):
        self.manager = manager
        self.motors = []
        self.motors.append(Motor(1,self.manager,40,2.0,1.0,1.0,MotorType.X8_90,10))
        self.motors.append(Motor(3,self.manager,40,2.0,1.0,1.0,MotorType.X8_90,10))
        self.motors.append(Motor(5,self.manager,25,1.5,1.0,1.0,MotorType.X4_24,10))
        self.motors.append(Motor(7,self.manager,25,1.5,1.0,1.0,MotorType.X4_24,10))
        self.motors.append(Motor(9,self.manager,25,1.5,1.0,1.0,MotorType.X4_24,10))
        self.motors.append(Motor(11,self.manager,25,1.5,1.0,1.0,MotorType.X4_24,10))

        self.motors.append(Motor(2,self.manager,40,2.0,1.0,1.0,MotorType.X8_90,10))
        self.motors.append(Motor(4,self.manager,40,2.0,1.0,1.0,MotorType.X8_90,10))
        self.motors.append(Motor(6,self.manager,25,1.5,1.0,1.0,MotorType.X4_24,10))
        self.motors.append(Motor(8,self.manager,25,1.5,1.0,1.0,MotorType.X4_24,10))
        self.motors.append(Motor(10,self.manager,25,1.5,1.0,1.0,MotorType.X4_24,10))
        self.motors.append(Motor(12,self.manager,25,1.5,1.0,1.0,MotorType.X4_24,10))


    def read_angles(self):
        output_angles = np.zeros(12, dtype=np.float32)
        for i in range(12):
            with self.motors[i]._state_lock:
                output_angles[i] = self.motors[i]._st.position
        return self.remap_motor_response(output_angles)
    
    def read_velocity(self):
        output_velocity = np.zeros(12, dtype=np.float32)
        for i in range(12):
            with self.motors[i]._state_lock:
                output_velocity[i] = self.motors[i]._st.velocity
        return self.remap_motor_response(output_velocity)
    def read_torque(self):
        output_torque = np.zeros(12, dtype=np.float32)
        for i in range(12):
            with self.motors[i]._state_lock:
                output_torque[i] = self.motors[i]._st.torque
        return self.remap_motor_response(output_torque)
    
    def read_data(self):
        output_angles = np.zeros(len(self.motors), dtype=np.float32)
        output_velocity = np.zeros(len(self.motors), dtype=np.float32)
        output_torque = np.zeros(len(self.motors), dtype=np.float32)
        motor_times = np.zeros(len(self.motors), dtype=np.uint64)
        for i in range(len(self.motors)):
            with self.motors[i]._state_lock:
                output_angles[i] = self.motors[i]._st.position
                output_velocity[i] = self.motors[i]._st.velocity
                output_torque[i] = self.motors[i]._st.torque
                motor_times[i] = self.motors[i]._st.last_update_ns
        return KinematicsData(
            angles=output_angles,
            velocity=output_velocity,
            torque=output_torque,
            motor_times=motor_times
        )


    def query_angles(self):
        for i in range(len(self.motors)):
            self.motors[i].query_position()
    
    def query_velocity(self):
        for i in range(len(self.motors)):
            self.motors[i].query_velocity()
    
    def query_torque(self):
        for i in range(len(self.motors)):
            self.motors[i].query_torque()
    
    def query_everything(self):
        self.query_angles()
        self.query_velocity()
        self.query_torque()

    def set_angles(self, angles=None):
        for i in range(len(self.motors)):
            # self.motors[i].set_pid_position(angles[i])
            self.motors[i].set_direct_position(radians = angles[i], speed=15)