from quadruped_leg_kinematics.param_helper import ParamLegs
import numpy as np
import time 
from param_logger import ParamLogger

default_pose = np.array([
    0.0,   
    0.7,   
    -1.3, 
])  # Default pose for the robot

logger = ParamLogger("jump_logs")
param = ParamLegs()
param.start()
time.sleep(2)
kinematics_data = param.get_kinematics_data()
qpos = kinematics_data.angles
# diff = default_pose - qpos
# for i in range(50): 
#     ctrl =diff*(i/50) + qpos 
ctrl = default_pose
#param.set_ctrl(ctrl)
time.sleep(0.02)  # Adjust the sleep time as needed

while True:
    kinematics_data = param.get_kinematics_data()
    qpos = kinematics_data.angles
    print(f"Current joint angles: {qpos}")
    print(f"Current current: {kinematics_data.motor_temp}")
    logger.log(None, None, kinematics_data.angles, None, kinematics_data.torque, kinematics_data.motor_current, None, None)
    time.sleep(0.01)  # Adjust the sleep time as needed
