from ga_one.ga_one_arms.kinematics.ga_one_arms import GaOneArms
import numpy as np
import time 


default_pose = np.array([
    0.0,   # Left
    0.0,   
    0.0,
    0.0,  
    0.0, 
    0.0,   
    0.0,  # Right
    0.0,  
    0,   
    0,  
    0.0,  
    0.0,   
])  # Default pose for the robot

ga_one = GaOneArms()
ga_one.start()
time.sleep(2)
kinematics_data = ga_one.get_kinematics_data()
qpos = kinematics_data.angles
# diff = default_pose - qpos
# for i in range(50): 
#     ctrl =diff*(i/50) + qpos 
ctrl = default_pose
ga_one.set_ctrl(ctrl)
time.sleep(0.02)  # Adjust the sleep time as needed

while True:
    kinematics_data = ga_one.get_kinematics_data()
    qpos = kinematics_data.angles
    print(f"Current joint angles: {qpos}")
    time.sleep(0.1)  # Adjust the sleep time as needed
