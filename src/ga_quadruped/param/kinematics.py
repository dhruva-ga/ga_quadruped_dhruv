from ga_can.devices.motor.motor import Motor,MotorType
import numpy as np
from dataclasses import dataclass

@dataclass
class KinematicsData:
    angles: np.ndarray
    velocity: np.ndarray
    torque: np.ndarray
    motor_times: np.ndarray  # Added to store motor times
    motor_current: np.ndarray
    motor_temp: np.ndarray
    motor_name: list

class ParamLegsKinematics:
    def __init__(self,manager):
        self.manager = manager
        self.motors = []

        x8_90_hip_stiffness = 25.0
        x8_90_hip_damping = 0.5

        x8_90_thigh_stiffness = 25.0
        x8_90_thigh_damping = 0.5

        x12_150_calf_stiffness = 25
        x12_150_calf_damping = 0.5

        # Front Left Leg
        self.motors.append(Motor(1,"FL_hip", self.manager,x8_90_hip_stiffness,x8_90_hip_damping,1.0,1.0,MotorType.X8_90,50))
        self.motors.append(Motor(2,"FL_thigh", self.manager,x8_90_thigh_stiffness,x8_90_thigh_damping,1.0,1.0,MotorType.X8_90,50))
        self.motors.append(Motor(3,"FL_calf", self.manager,x12_150_calf_stiffness,x12_150_calf_damping,1.0,1.0,MotorType.X12_150,50))

        # Front Right Leg
        self.motors.append(Motor(4,"FR_hip", self.manager,x8_90_hip_stiffness,x8_90_hip_damping,1.0,1.0,MotorType.X8_90,50))
        self.motors.append(Motor(5,"FR_thigh", self.manager,x8_90_thigh_stiffness,x8_90_thigh_damping,1.0,1.0,MotorType.X8_90,50))
        self.motors.append(Motor(6,"FR_calf", self.manager,x12_150_calf_stiffness,x12_150_calf_damping,1.0,1.0,MotorType.X12_150,50))

        # Rear Left Leg
        self.motors.append(Motor(7,"RL_hip", self.manager,x8_90_hip_stiffness,x8_90_hip_damping,1.0,1.0,MotorType.X8_90,50))
        self.motors.append(Motor(8,"RL_thigh", self.manager,x8_90_thigh_stiffness,x8_90_thigh_damping,1.0,1.0,MotorType.X8_90,50))
        self.motors.append(Motor(9,"RL_calf", self.manager,x12_150_calf_stiffness,x12_150_calf_damping,1.0,1.0,MotorType.X12_150,50))
        
        # Rear Right Leg
        self.motors.append(Motor(10,"RR_hip", self.manager,x8_90_hip_stiffness,x8_90_hip_damping,1.0,1.0,MotorType.X8_90,50))
        self.motors.append(Motor(11,"RR_thigh", self.manager,x8_90_thigh_stiffness,x8_90_thigh_damping,1.0,1.0,MotorType.X8_90,50))
        self.motors.append(Motor(12,"RR_calf", self.manager,x12_150_calf_stiffness,x12_150_calf_damping,1.0,1.0,MotorType.X12_150,50))
        
        self.motor_name = [motor.motor_name for motor in self.motors]

    def remap_motor_response(self, data):
        # no remap for now
        return data

    def read_angles(self):
        output_angles = np.zeros(len(self.motors), dtype=np.float32)
        for i in range(len(self.motors)):
            with self.motors[i]._state_lock:
                output_angles[i] = self.motors[i]._st.position
        return self.remap_motor_response(output_angles)
    
    def read_velocity(self):
        output_velocity = np.zeros(len(self.motors), dtype=np.float32)
        for i in range(len(self.motors)):
            with self.motors[i]._state_lock:
                output_velocity[i] = self.motors[i]._st.velocity
        return self.remap_motor_response(output_velocity)
    
    def read_torque(self):
        output_torque = np.zeros(len(self.motors), dtype=np.float32)
        for i in range(len(self.motors)):
            with self.motors[i]._state_lock:
                output_torque[i] = self.motors[i]._st.torque
        return self.remap_motor_response(output_torque)

    def read_data(self):
        output_angles = np.zeros(len(self.motors), dtype=np.float32)
        output_velocity = np.zeros(len(self.motors), dtype=np.float32)
        output_torque = np.zeros(len(self.motors), dtype=np.float32)
        motor_times = np.zeros(len(self.motors), dtype=np.uint64)
        motor_current = np.zeros(len(self.motors))
        motor_temp = np.zeros(len(self.motors), dtype=np.uint64)
        for i in range(len(self.motors)):
            try:
                with self.motors[i]._state_lock:
                    output_angles[i] = self.motors[i]._st.position
                    output_velocity[i] = self.motors[i]._st.velocity
                    output_torque[i] = self.motors[i]._st.torque
                    motor_times[i] = self.motors[i]._st.last_update_ns
                    motor_current[i] = self.motors[i]._st.current
                    motor_temp[i] = self.motors[i]._st.temperature
            except TypeError as e:
                print("Error in getting value")
        return KinematicsData(
            angles=output_angles,
            velocity=output_velocity,
            torque=output_torque,
            motor_times=motor_times,
            motor_current=motor_current,
            motor_temp=motor_temp,
            motor_name=self.motor_name
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
            self.motors[i].set_pid_position(angles[i])
            # self.motors[i].set_direct_position(radians = angles[i], speed=15)